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
    """

    def __init__(
        self,
        diversity_op: Callable[[Solution, Solution], float],
        params: PopulationParams | None = None,
        custom_selector: Optional[Path] = None,
        custom_survivor: Optional[Path] = None,  # New parameter
        selector_timeout: float = 10.0,
    ):
        self._op = diversity_op
        self._params = params if params is not None else PopulationParams()
        self.selector_timeout = selector_timeout

        self._selector = self._default_selector
        if custom_selector:
            self._selector = self._load_custom_selector(custom_selector)

        # Survivor selection
        self._survivor_selector = None
        if custom_survivor:
            self._survivor_selector = self._load_custom_survivor(custom_survivor)

        self._feas = SubPopulation(diversity_op, self._params)
        self._infeas = SubPopulation(diversity_op, self._params)
        

    def _load_custom_selector(self, path: Path) -> Callable:
        module = self._safe_import_with_timeout(path)
        if module is None:
            raise ValueError(f"Failed to import module from {path}")
        if not hasattr(module, "select_parents"):
            raise ValueError("LLM module must define 'select_parents'")
        func = module.select_parents
        self._validate_signature(func, ["population", "rng", "cost_evaluator", "k"])
        return func

    def _load_custom_survivor(self, path: Path) -> Callable:
        module = self._safe_import_with_timeout(path)
        if module is None:
            raise ValueError(f"Failed to import module from {path}")
        if not hasattr(module, "select_survivors"):
            raise ValueError("LLM module must define 'select_survivors'")
        func = module.select_survivors
        self._validate_signature(func, ["population", "cost_evaluator", "params"])
        return func

    def _validate_signature(self, func: Callable, expected_names: list[str]):
        sig = inspect.signature(func)
        if list(sig.parameters.keys()) != expected_names:
            raise ValueError(f"Invalid signature for {func.__name__}, expected {expected_names}")

    def _extract_python_code(self, raw_code: str) -> str:
        """
        Extract Python code from ```python ... ``` block and remove example usage.
        More lenient approach - keeps more code structures.
        """
        import re
        import ast
    
        # Step 1: Extract code block
        match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
        code = match.group(1).strip() if match else raw_code
    
        try:
            # Step 2: Parse and filter AST - be more permissive
            tree = ast.parse(code)
            filtered_nodes = []
        
            for node in tree.body:
                if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
                    filtered_nodes.append(node)
                elif isinstance(node, ast.Assign):
                    # Keep top-level assignments like constants
                    filtered_nodes.append(node)
                elif isinstance(node, ast.If):
                    # Keep if statements that might contain function definitions
                    # but skip if __name__ == "__main__" blocks
                    if not (isinstance(node.test, ast.Compare) and 
                           isinstance(node.test.left, ast.Name) and 
                           node.test.left.id == "__name__"):
                        filtered_nodes.append(node)
                # Skip: Expr(Call(...)), if __name__ == "__main__", etc.
        
            if not filtered_nodes:
                # If filtering results in empty code, return original
                return code
                
            new_tree = ast.Module(body=filtered_nodes, type_ignores=[])
            return ast.unparse(new_tree)  # Requires Python 3.9+
            
        except (SyntaxError, ValueError) as e:
            print(f"[Population] AST parsing failed, using raw code: {e}")
            return code

    def _safe_import_with_timeout(
        self,
        path: Path,
        timeout: float = 10.0,
        restricted_globals: dict | None = None,
    ) -> ModuleType | None:
        import multiprocessing
        import tempfile
        import importlib.util
    
        try:
            raw_text = path.read_text()
        except Exception as e:
            print(f"[Population] Failed to read file {path}: {e}")
            return None
            
        code = self._extract_python_code(raw_text)
        
        # First, validate the code in a separate process
        def validate_worker(code_str: str, queue):
            try:
                # Try to compile the code to check for syntax errors
                compile(code_str, '<string>', 'exec')
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
            print(f"[Population] Code validation timed out for {path}")
            return None
            
        if queue.empty():
            print(f"[Population] Code validation process crashed for {path}")
            return None
            
        try:
            status, result = queue.get()
            if status != "valid":
                print(f"[Population] Invalid code in {path}: {result}")
                return None
        except Exception as e:
            print(f"[Population] Failed to get validation result: {e}")
            return None
        
        # If validation passed, import the module directly (no multiprocessing)
        tmp_path = None
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tmp:
                tmp.write(code)
                tmp_path = tmp.name

            # Load the module
            spec = importlib.util.spec_from_file_location("llm_module", tmp_path)
            if spec is None or spec.loader is None:
                print("[Population] Failed to load spec")
                return None

            module = importlib.util.module_from_spec(spec)

            # Enhanced restricted globals
            if restricted_globals is None:
                restricted_globals = {
                    # Built-ins
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
                    "print": print,  # For debugging
                    
                    # Math and random
                    "random": random,
                    "np": np,
                    "numpy": np,
                
                    # Typing helpers
                    "Tuple": Tuple,
                    "Callable": Callable,
                    "Iterator": Iterator,
                    "overload": overload,
                
                    # All PyVRP types / classes from _pyvrp.pyi
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

            # Inject restricted globals
            for key, value in restricted_globals.items():
                setattr(module, key, value)

            # Execute the module
            spec.loader.exec_module(module)
            return module

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"[Population] Import failed for {path}: {error_msg}")
            return None
        finally:
            # Clean up temporary file
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except:
                    pass
    
        queue = multiprocessing.Queue()

        # Enhanced restricted globals
        if restricted_globals is None:
            restricted_globals = {
                # Built-ins
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
                "print": print,  # For debugging
                
                # Math and random
                "random": random,
                "np": np,
                "numpy": np,
            
                # Typing helpers
                "Tuple": Tuple,
                "Callable": Callable,
                "Iterator": Iterator,
                "overload": overload,
            
                # All PyVRP types / classes from _pyvrp.pyi
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
    
        proc = multiprocessing.Process(
            target=worker, args=(code, restricted_globals, queue)
        )
        proc.start()
        proc.join(timeout)
    
        if proc.is_alive():
            proc.terminate()
            proc.join()
            print(f"[Population] Import timed out for {path}")
            return None
    
        if queue.empty():
            print(f"[Population] Import process crashed silently for {path}")
            return None
    
        try:
            status, result = queue.get()
            if status == "ok":
                return result
            else:
                print(f"[Population] Import failed for {path}: {result}")
                return None
        except Exception as e:
            print(f"[Population] Failed to get result from queue: {e}")
            return None

    def __iter__(self) -> Generator[Solution, None, None]:
        """
        Iterates over the solutions contained in this population.
        """
        for item in self._feas:
            yield item.solution

        for item in self._infeas:
            yield item.solution

    def __len__(self) -> int:
        """
        Returns the current population size.
        """
        return len(self._feas) + len(self._infeas)

    def _update_fitness(self, cost_evaluator: CostEvaluator):
        """
        Updates the biased fitness values for the subpopulations.

        Parameters
        ----------
        cost_evaluator
            CostEvaluator to use for computing the fitness.
        """
        self._feas.update_fitness(cost_evaluator)
        self._infeas.update_fitness(cost_evaluator)

    def num_feasible(self) -> int:
        """
        Returns the number of feasible solutions in the population.
        """
        return len(self._feas)

    def num_infeasible(self) -> int:
        """
        Returns the number of infeasible solutions in the population.
        """
        return len(self._infeas)

    def add(self, solution: Solution, cost_evaluator: CostEvaluator):
        """
        Add solution; if custom_survivor defined and overflow, call LLM logic.
        """
        subpop = self._feas if solution.is_feasible() else self._infeas

        # if overflow and custom survivor selection, override C++ purge
        if self._survivor_selector and len(subpop) >= self._params.max_pop_size:
            # extract all solutions
            solutions = [item.solution for item in subpop]
            # run LLM survivor logic & enforce limit
            survivors = self._survivor_selector(
                solutions,
                cost_evaluator,
                self._params
            )[:self._params.max_pop_size]
            
            # Create NEW subpopulation
            new_subpop = SubPopulation(self._op, self._params)
            for sol in survivors:
                new_subpop.add(sol, cost_evaluator)

            # Replace reference
            if solution.is_feasible():
                self._feas = new_subpop
            else:
                self._infeas = new_subpop
        else:
            subpop.add(solution, cost_evaluator)
    
    def clear(self):
        """
        Clears the population by removing all solutions currently in the
        population.
        """
        self._feas = SubPopulation(self._op, self._params)
        self._infeas = SubPopulation(self._op, self._params)

    def tournament(
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ) -> Solution:
        """
        Selects a solution from this population by k-ary tournament, based
        on the (internal) fitness values of the selected solutions.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        Solution
            The selected solution.
        """
        self._update_fitness(cost_evaluator)
        return self._tournament(rng, k)

    def _tournament(self, rng: RandomNumberGenerator, k: int) -> Solution:
        if k <= 0:
            raise ValueError(f"Expected k > 0; got k = {k}.")

        def select():
            num_feas = len(self._feas)
            idx = rng.randint(len(self))

            if idx < num_feas:
                return self._feas[idx]

            return self._infeas[idx - num_feas]

        items = [select() for _ in range(k)]
        fittest = min(items, key=lambda item: item.fitness)
        return fittest.solution

    def _run_selector_safe(
        self,
        population: list[Solution],
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int,
        queue: Queue
    ):
        try:
            res = self._selector(population, rng, cost_evaluator, k)
            queue.put((True, res))
        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            queue.put((False, error_msg))
        
    def _default_selector(
        self, 
        rng: RandomNumberGenerator, 
        cost_evaluator: CostEvaluator, 
        k: int = 2
    ) -> tuple[Solution, Solution]:
        """Original tournament selection logic."""
        """
        Selects two (if possible non-identical) parents by tournament, subject
        to a diversity restriction.

        Parameters
        ----------
        rng
            Random number generator.
        cost_evaluator
            Cost evaluator to use when computing the fitness.
        k
            The number of solutions to draw for the tournament. Defaults to
            two, which results in a binary tournament.

        Returns
        -------
        tuple
            A solution pair (parents).
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
        self,
        rng: RandomNumberGenerator,
        cost_evaluator: CostEvaluator,
        k: int = 2,
    ) -> tuple[Solution, Solution]:
        """Main entry point for parent selection."""
        if self._selector == self._default_selector:
            return self._default_selector(rng, cost_evaluator, k)
        
        # For LLM selector: combine feasible/infeasible populations
        population = list(self)
        queue: Queue = Queue()
        proc = Process(
            target=self._run_selector_safe,
            args=(population, rng, cost_evaluator, k, queue),
        )
        proc.start()
        proc.join(self.selector_timeout)

        if proc.is_alive():
            proc.terminate()
            proc.join()
            return self._default_selector(rng, cost_evaluator, k)

        if queue.empty():
            return self._default_selector(rng, cost_evaluator, k)

        success, result = queue.get()
        if not success or not isinstance(result, tuple) or len(result) != 2:
            print(f"[Population] LLM selector failed: {result}")
            return self._default_selector(rng, cost_evaluator, k)

        return result
