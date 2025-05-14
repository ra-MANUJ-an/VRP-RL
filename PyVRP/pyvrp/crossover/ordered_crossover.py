# File: pyvrp/crossover/ordered_crossover.py (modified)
from __future__ import annotations

import ast
import importlib
import inspect
import sys
import random
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Callable, Generator, Optional

from pyvrp._pyvrp import (
    CostEvaluator,
    ProblemData,
    RandomNumberGenerator,
    Solution,
)
from pyvrp.crossover._crossover import ordered_crossover as _ox

def _extract_python_code(raw_code: str) -> str:
    """
    Extract Python code from ```python ... ``` block and remove example usage.
    Keeps only class/function definitions and top-level imports.
    """
    import re
    import ast
    
    # Step 1: Extract code block
    match = re.search(r"```python(.*?)```", raw_code, re.DOTALL)
    code = match.group(1).strip() if match else raw_code

    # Step 2: Parse and filter AST
    tree = ast.parse(code)
    filtered_nodes = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)):
            filtered_nodes.append(node)
        elif isinstance(node, ast.Assign):
            # Optionally keep top-level assignments like constants
            filtered_nodes.append(node)
        # skip: Expr(Call(...)), if __name__ == "__main__", etc.

    new_tree = ast.Module(body=filtered_nodes, type_ignores=[])
    return ast.unparse(new_tree)  # Requires Python 3.9+

def _validate_ast(code: str):
    """Ensure code contains only allowed constructs."""
    tree = ast.parse(code)

    allowed_builtins = (ast.Name, ast.Attribute, ast.Subscript)
    prohibited = (
        ast.Import, ast.ImportFrom,
    )
    
    for node in ast.walk(tree):
        if isinstance(node, prohibited):
            raise ValueError(f"Prohibited construct: {type(node).__name__}")

def _safe_import(path: Path) -> ModuleType:
    """Safely import module with restricted environment."""
    raw_text = path.read_text()
    code = _extract_python_code(raw_text)  # Extract clean Python code

    # _validate_ast(code)  # Validate cleaned code

    spec = importlib.util.spec_from_file_location("llm_ox", path)
    module = importlib.util.module_from_spec(spec)
    
    restricted_globals = {
        "__builtins__": __builtins__,  # Allow standard Python builtins
        "Solution": Solution,
        "CostEvaluator": CostEvaluator,
        "RandomNumberGenerator": RandomNumberGenerator,
    }

    module.__dict__.update(restricted_globals)

    # Instead of using spec.loader.exec_module(), run our extracted code
    exec(compile(code, filename=str(path), mode="exec"), module.__dict__)
    return module

# def _safe_import(path: Path) -> ModuleType:
#     """Safely import module with restricted environment."""
#     raw_text = path.read_text()
#     code = _extract_python_code(raw_text)  # Extract clean Python code

#     _validate_ast(code)  # Validate cleaned code

#     spec = importlib.util.spec_from_file_location("llm_ox", path)
#     module = importlib.util.module_from_spec(spec)
    
#     restricted_globals = {
#         "__builtins__": {
#             "abs": abs,
#             "min": min,
#             "max": max,
#             "len": len,
#             "range": range,
#             "sorted": sorted,
#             "list": list,
#             "tuple": tuple,
#             "type": type,
#             "int": int,
#             "Exception": Exception,
#             "ValueError": ValueError,
#             "random": random,
#         },
#         "Solution": Solution,
#         "CostEvaluator": CostEvaluator,
#         "RandomNumberGenerator": RandomNumberGenerator,
#     }

#     module.__dict__.update(restricted_globals)

#     # Instead of using spec.loader.exec_module(), run our extracted code
#     exec(compile(code, filename=str(path), mode="exec"), module.__dict__)
#     return module
    
def _validate_signature(func: Callable):
    """Ensure function has correct signature."""
    sig = inspect.signature(func)
    expected = [
        "parents",
        "data",
        "indices"
    ]

    if list(sig.parameters.keys()) != expected:
        raise ValueError(f"Invalid signature. Expected {expected}")


def _load_custom_crossover(path: Path) -> Optional[Callable]:
    """Load and validate LLM-generated selection function."""
    module = _safe_import(path)

    if not hasattr(module, "ordered_crossover_op"):
        raise ValueError("LLM module must contain 'ordered_crossover_op' function")

    func = module.ordered_crossover_op
    _validate_signature(func)
    return func


def ordered_crossover(
    parents: tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    rng: RandomNumberGenerator,
    *,
    custom_path: Optional[Path] = None,  # New parameter
) -> Solution:
    """
    Performs an ordered crossover (OX) operation between the two given parents.
    The clients between two randomly selected indices of the first route are
    copied into a new solution, and any missing clients that are present in the
    second route are then copied in as well. See [1]_ for details.

    .. warning::

       This operator explicitly assumes the problem instance is a TSP. You
       should use a different crossover operator if that is not the case.

    Parameters
    ----------
    parents
        The two parent solutions to create an offspring from.
    data
        The problem instance.
    cost_evaluator
        Cost evaluator object. Unused by this operator.
    rng
        The random number generator to use.

    Returns
    -------
    Solution
        A new offspring.

    Raises
    ------
    ValueError
        When the given data instance is not a TSP, particularly, when there is
        more than one vehicle in the data.

    References
    ----------
    .. [1] I. M. Oliver, D. J. Smith, and J. R. C. Holland. 1987. A study of
           permutation crossover operators on the traveling salesman problem.
           In *Proceedings of the Second International Conference on Genetic
           Algorithms on Genetic algorithms and their application*. 224 - 230.
    """
    """
    Full implementation with proper fallback to original C++ logic
    """

    if data.num_vehicles != 1:
        msg = f"Expected a TSP, got {data.num_vehicles} vehicles instead."
        raise ValueError(msg)

    first, second = parents

    if first.num_clients() == 0:
        return second

    if second.num_clients() == 0:
        return first

    # Generate [start, end) indices in the route of the first parent solution.
    # If end < start, the index segment wraps around. Clients in this index
    # segment are copied verbatim into the offspring solution.
    first_route = first.routes()[0]
    start = rng.randint(len(first_route))
    end = rng.randint(len(first_route))

    # When start == end we try to find a different end index, such that the
    # offspring actually inherits something from each parent.
    while start == end and len(first_route) > 1:
        end = rng.randint(len(first_route))

    # # Attempt LLM-generated crossover first
    # if custom_path:
    #     try:
    #         custom = _load_custom_crossover(custom_path)
    #         offspring = custom(parents, data, (start, end))
    #         # sanity check
    #         if not isinstance(offspring, Solution):
    #             raise ValueError(f"Custom returned {type(offspring)}, expected Solution")
    #         return offspring
    #     except Exception as e:
    #         print(f"Custom OX failed: {e}")

    # # Attempt LLM-generated crossover first
    # if custom_path:
    #     try:
    #         custom = _load_custom_crossover(custom_path)
    #         return custom(parents, data, (start, end)) # (parents, data, indices)
    #     except Exception as e:
    #         print(f"Custom OX failed: {e}")

    # Attempt LLM-generated crossover first
    if custom_path:
        custom = _load_custom_crossover(custom_path)
        return custom(parents, data, (start, end)) # (parents, data, indices)

    # print("OXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    # Original C++ fallback with all safety checks
    return _ox(parents, data, (start, end))