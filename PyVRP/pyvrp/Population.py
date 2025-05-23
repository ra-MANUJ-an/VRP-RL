from __future__ import annotations

import ast
import importlib
import inspect
import sys
import random
import os
import tempfile
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator, Optional
from typing import Tuple, Callable, Iterator, overload
from multiprocessing import Process, Queue

from pyvrp._pyvrp import PopulationParams, SubPopulation

# if TYPE_CHECKING:
from pyvrp._pyvrp import CostEvaluator, RandomNumberGenerator, Solution

import random
import numpy as np
from typing import Tuple, Callable, Iterator, overload

from pyvrp._pyvrp import (
    CostEvaluator,
    DynamicBitset,
    Client,
    ClientGroup,
    Depot,
    VehicleType,
    ProblemData,
    ScheduledVisit,
    Route,
    Solution,
    PopulationParams,
    SubPopulation,
    SubPopulationItem,
    DistanceSegment,
    LoadSegment,
    DurationSegment,
    RandomNumberGenerator,
)


class Population:
    """
    Creates a Population instance.

    Parameters
    ----------
    diversity_op
        Operator to use to determine pairwise diversity between solutions. Have
        a look at :mod:`pyvrp.diversity` for available operators.
    params
        Population parameters. If not provided, a default will be used.
    custom_selector
        Path to Python module containing LLM-generated selection logic.
    custom_survivor
        Path to Python module containing LLM-generated survivor logic.
    selector_timeout
        Seconds to wait before raising an error if the LLM takes too long.
    """

    def __init__(
        self,
        diversity_op: Callable[[Solution, Solution], float],
        params: PopulationParams | None = None,
        custom_selector: Optional[Path] = None,
        custom_survivor: Optional[Path] = None,
        selector_timeout: float = 10.0,
    ):
        self._op = diversity_op
        self._params = params if params is not None else PopulationParams()
        self.selector_timeout = selector_timeout

        # By default, use the built-in tournament selector
        self._selector = self._default_selector
        if custom_selector:
            try:
                self._selector = self._load_custom_selector(custom_selector)
            except ValueError as e:
                # Fail fast if the custom selector cannot be imported
                raise ValueError(f"Failed to import custom selector: {e}")

        # By default, no survivor logic (so we let C++ purge). If the user
        # provided a path, try to load it—and fail fast on error.
        self._survivor_selector = None
        if custom_survivor:
            try:
                self._survivor_selector = self._load_custom_survivor(custom_survivor)
            except ValueError as e:
                raise ValueError(f"Failed to import custom survivor: {e}")

        # Two subpopulations, one for feasible, one for infeasible
        self._feas = SubPopulation(diversity_op, self._params)
        self._infeas = SubPopulation(diversity_op, self._params)

    def _load_custom_selector(self, path: Path) -> Callable:
        module = self._safe_import_with_timeout(path)
        if module is None:
            raise ValueError(f"Could not load module from {path}")
        if not hasattr(module, "select_parents"):
            raise ValueError("LLM selector module must define `select_parents`")
        func = module.select_parents
        self._validate_signature(func, ["population", "rng", "cost_evaluator", "k"])
        return func

    def _load_custom_survivor(self, path: Path) -> Callable:
        module = self._safe_import_with_timeout(path)
        if module is None:
            raise ValueError(f"Could not load module from {path}")
        if not hasattr(module, "select_survivors"):
            raise ValueError("LLM survivor module must define `select_survivors`")
        func = module.select_survivors
        self._validate_signature(func, ["population", "cost_evaluator", "params"])
        return func

    def _validate_signature(self, func: Callable, expected_names: list[str]):
        sig = inspect.signature(func)
        if list(sig.parameters.keys()) != expected_names:
            raise ValueError(
                f"Invalid signature for `{func.__name__}`; expected parameters {expected_names}"
            )

    def _extract_python_code(self, raw_code: str) -> str:
        """
        Extract Python code from ```python ... ``` blocks and drop everything else
        except top‐level class/function definitions, imports, and assignments.
        """
        import re

        match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
        code = match.group(1).strip() if match else raw_code

        tree = ast.parse(code)
        filtered_nodes = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                filtered_nodes.append(node)
            elif isinstance(node, ast.Assign):
                filtered_nodes.append(node)
            # drop everything else (e.g. if __name__ == "__main__", Expr(Call(...)), etc.)

        new_tree = ast.Module(body=filtered_nodes, type_ignores=[])
        return ast.unparse(new_tree)  # Python 3.9+ required

    def _safe_import_with_timeout(
        self,
        path: Path,
        timeout: float = 10.0,
        restricted_globals: dict | None = None,
    ) -> ModuleType | None:
        """
        1. Read the file at `path`.
        2. Extract only the top‐level defs/imports/assigns via `_extract_python_code`.
        3. Spawn a helper process to compile(...) that snippet—enforcing `timeout`.
           If it times out or has syntax errors, return None.
        4. If compile is valid, write to a temp file and then `exec_module(...)` it
           in the main process. But before exec’ing, *temporarily* insert
           `pyvrp._pyvrp` into `sys.modules['pyvrp']` so that “from pyvrp import X”
           works correctly.
        """
        import multiprocessing

        # Step A: Read the raw text
        try:
            raw_text = path.read_text()
        except Exception as e:
            print(f"[Population] Failed to read `{path}`: {e}")
            return None

        # Step B: Filter out everything except top‐level defs/imports/assigns
        code = self._extract_python_code(raw_text)

        # Step C: Spawn a subprocess to do a syntax check via compile(...):
        def validate_worker(code_str: str, queue: Queue):
            try:
                compile(code_str, "<string>", "exec")
                queue.put(("valid", True))
            except Exception as e:
                queue.put(("invalid", str(e)))

        queue = multiprocessing.Queue()
        proc = multiprocessing.Process(target=validate_worker, args=(code, queue))
        proc.start()
        proc.join(timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            print(f"[Population] Code validation timed out for `{path}`")
            return None

        if queue.empty():
            print(f"[Population] Code validation subprocess crashed for `{path}`")
            return None

        status, payload = queue.get()
        if status != "valid":
            print(f"[Population] Syntax error in `{path}`: {payload}")
            return None

        # Step D: Write the filtered code to a temporary file and import it
        tmp_module_path = None
        try:
            rand_suffix = random.randrange(0, 10**8)
            module_name = f"llm_module_{rand_suffix}"
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_module_path = tmp.name

            spec = importlib.util.spec_from_file_location(module_name, tmp_module_path)
            if spec is None or spec.loader is None:
                print(f"[Population] Failed to load spec from `{tmp_module_path}`")
                return None

            module = importlib.util.module_from_spec(spec)

            # Build a restricted global namespace if none provided
            if restricted_globals is None:
                restricted_globals = {
                    "__builtins__": __builtins__,
                    "__import__": __import__,
                    "min": min,
                    "max": max,
                    "len": len,
                    "range": range,
                    "sorted": sorted,
                    "Exception": Exception,
                    "list": list,
                    "tuple": tuple,
                    "dict": dict,
                    "set": set,
                    "type": type,
                    "int": int,
                    "float": float,
                    "str": str,
                    "bool": bool,
                    "abs": abs,
                    "sum": sum,
                    "any": any,
                    "all": all,
                    "enumerate": enumerate,
                    "zip": zip,
                    "print": print,  # for debugging
                    # math/random
                    "random": random,
                    "np": np,
                    "numpy": np,
                    # typing helpers
                    "Tuple": Tuple,
                    "Callable": Callable,
                    "Iterator": Iterator,
                    "overload": overload,
                    # all core symbols from pyvrp._pyvrp
                    "CostEvaluator": CostEvaluator,
                    "DynamicBitset": DynamicBitset,
                    "Client": Client,
                    "ClientGroup": ClientGroup,
                    "Depot": Depot,
                    "VehicleType": VehicleType,
                    "ProblemData": ProblemData,
                    "ScheduledVisit": ScheduledVisit,
                    "Route": Route,
                    "Solution": Solution,
                    "PopulationParams": PopulationParams,
                    "SubPopulation": SubPopulation,
                    "SubPopulationItem": SubPopulationItem,
                    "DistanceSegment": DistanceSegment,
                    "LoadSegment": LoadSegment,
                    "DurationSegment": DurationSegment,
                    "RandomNumberGenerator": RandomNumberGenerator,
                }

            # Inject these into the new module’s namespace
            for key, val in restricted_globals.items():
                setattr(module, key, val)

            # *** TRICK: temporarily make "pyvrp" point to pyvrp._pyvrp ***
            # so that user code like “from pyvrp import DistanceSegment” works.
            prev_pyvrp = sys.modules.get("pyvrp")
            try:
                sys.modules["pyvrp"] = importlib.import_module("pyvrp._pyvrp")
                spec.loader.exec_module(module)
            finally:
                # restore whatever was in sys.modules["pyvrp"] before
                if prev_pyvrp is None:
                    del sys.modules["pyvrp"]
                else:
                    sys.modules["pyvrp"] = prev_pyvrp

            return module

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            print(f"[Population] Failed to import `{path}`: {e}\n{tb}")
            return None

        finally:
            if tmp_module_path and os.path.exists(tmp_module_path):
                try:
                    os.unlink(tmp_module_path)
                except OSError:
                    pass

    def __iter__(self) -> Generator[Solution, None, None]:
        for item in self._feas:
            yield item.solution
        for item in self._infeas:
            yield item.solution

    def __len__(self) -> int:
        return len(self._feas) + len(self._infeas)

    def _update_fitness(self, cost_evaluator: CostEvaluator):
        self._feas.update_fitness(cost_evaluator)
        self._infeas.update_fitness(cost_evaluator)

    def num_feasible(self) -> int:
        return len(self._feas)

    def num_infeasible(self) -> int:
        return len(self._infeas)

    def _run_survivor_safe(
        self,
        population: list[Solution],
        cost_evaluator: CostEvaluator,
        params: PopulationParams,
        queue: Queue,
    ):
        """
        Wrapped in a Process: call the LLM‐defined `select_survivors(...)`.
        If it throws or returns wrong, we send (False, error_msg). Otherwise (True, list_of_solutions).
        """
        import traceback

        try:
            survivors = self._survivor_selector(population, cost_evaluator, params)
            queue.put((True, survivors))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put((False, tb))

    def add(self, solution: Solution, cost_evaluator: CostEvaluator):
        """
        Add a solution. If custom_survivor is defined and subpopulation is full,
        call it via a subprocess with a timeout. If it times out or fails,
        raise a RuntimeError (no fallback to default C++ logic).
        """
        subpop = self._feas if solution.is_feasible() else self._infeas

        # If we have a custom survivor and the subpopulation is already full
        if self._survivor_selector and len(subpop) >= self._params.max_pop_size:
            # 1. Gather existing solutions + the new candidate
            existing_solutions = [item.solution for item in subpop]
            candidates = existing_solutions + [solution]

            # 2. Run LLM code in a subprocess with timeout
            queue: Queue = Queue()
            proc = Process(
                target=self._run_survivor_safe,
                args=(candidates, cost_evaluator, self._params, queue),
            )
            proc.start()
            proc.join(self.selector_timeout)

            timed_out = proc.is_alive()
            if timed_out:
                proc.terminate()
                proc.join()
                raise RuntimeError(
                    f"Custom survivor selection timed out after {self.selector_timeout} seconds"
                )

            if queue.empty():
                raise RuntimeError("Custom survivor selection process returned no output")

            success, payload = queue.get()
            if not success or not isinstance(payload, list):
                raise RuntimeError(f"Custom survivor selection failed: {payload}")

            survivors = payload[: self._params.max_pop_size]

            # 3. Rebuild subpopulation from survivors
            new_subpop = SubPopulation(self._op, self._params)
            for sol in survivors:
                new_subpop.add(sol, cost_evaluator)

            if solution.is_feasible():
                self._feas = new_subpop
            else:
                self._infeas = new_subpop

        else:
            # Normal behavior: let SubPopulation<>::add handle it
            subpop.add(solution, cost_evaluator)

    def clear(self):
        self._feas = SubPopulation(self._op, self._params)
        self._infeas = SubPopulation(self._op, self._params)

    def tournament(
        self, rng: RandomNumberGenerator, cost_evaluator: CostEvaluator, k: int = 2
    ) -> Solution:
        """
        Draw k items, pick the best (by fitness). Returns one Solution.
        """
        self._update_fitness(cost_evaluator)
        return self._tournament(rng, k)

    def _tournament(self, rng: RandomNumberGenerator, k: int) -> Solution:
        """
        Perform a k-way tournament. Raises if population is empty or k <= 0.
        """
        if len(self) == 0:
            raise RuntimeError("Cannot perform tournament on empty population")
        if k <= 0:
            raise ValueError(f"Expected k > 0; got k = {k}")

        def select_one():
            n_feas = len(self._feas)
            idx = rng.randint(len(self))
            if idx < n_feas:
                return self._feas[idx]
            return self._infeas[idx - n_feas]

        participants = [select_one() for _ in range(k)]
        winner = min(participants, key=lambda item: item.fitness)
        return winner.solution

    def _run_selector_safe(
        self,
        population: list[Solution],
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int,
        queue: Queue,
    ):
        """
        Wrapped in a Process: call the LLM-defined `select_parents(...)`.
        Send (True, (sol1, sol2)) or (False, error_msg).
        """
        import traceback

        try:
            result = self._selector(population, rng, cost_evaluator, k)
            queue.put((True, result))
        except Exception as e:
            tb = traceback.format_exc()
            queue.put((False, tb))

    def _default_selector(
        self, rng: RandomNumberGenerator, cost_evaluator: CostEvaluator, k: int = 2
    ) -> tuple[Solution, Solution]:
        """
        Exactly what PyVRP used to do: two successive tournaments, then enforce diversity.
        """
        self._update_fitness(cost_evaluator)
        first = self._tournament(rng, k)
        second = self._tournament(rng, k)

        diversity = self._op(first, second)
        lb = self._params.lb_diversity
        ub = self._params.ub_diversity

        tries = 1
        while not (lb <= diversity <= ub) and tries <= 10:
            tries += 1
            second = self._tournament(rng, k)
            diversity = self._op(first, second)

        return first, second

    def select(
        self, rng: RandomNumberGenerator, cost_evaluator: CostEvaluator, k: int = 2
    ) -> tuple[Solution, Solution]:
        """
        If no custom selector was provided, call _default_selector. Otherwise,
        run the LLM in a subprocess (timeout enforced), then validate the result.
        """
        if self._selector == self._default_selector:
            return self._default_selector(rng, cost_evaluator, k)

        if len(self) == 0:
            raise RuntimeError("Cannot select parents from an empty population")
        if k <= 0:
            raise ValueError(f"Expected k > 0; got k = {k}")

        population_list = list(self)
        queue: Queue = Queue()
        proc = Process(
            target=self._run_selector_safe,
            args=(population_list, rng, cost_evaluator, k, queue),
        )
        proc.start()
        proc.join(self.selector_timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            raise RuntimeError(f"Custom selector timed out after {self.selector_timeout} seconds")

        if queue.empty():
            raise RuntimeError("Custom selector process returned no output")

        success, payload = queue.get()
        if not success or not isinstance(payload, tuple) or len(payload) != 2:
            raise RuntimeError(f"Custom selector failed: {payload}")

        return payload
