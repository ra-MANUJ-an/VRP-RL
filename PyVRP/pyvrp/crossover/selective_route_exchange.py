from warnings import warn

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
from pyvrp.crossover._crossover import selective_route_exchange as _srex
from pyvrp.exceptions import TspWarning

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

    spec = importlib.util.spec_from_file_location("llm_srex", path)
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

#     spec = importlib.util.spec_from_file_location("llm_srex", path)
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
        "cost_evaluator",
        "start_indices",
        "num_moved_routes"
    ]

    if list(sig.parameters.keys()) != expected:
        raise ValueError(f"Invalid signature. Expected {expected}")


def _load_custom_crossover(path: Path) -> Optional[Callable]:
    """Load and validate LLM-generated selection function."""
    module = _safe_import(path)

    if not hasattr(module, "selective_route_exchange_op"):
        raise ValueError("LLM module must contain 'selective_route_exchange_op' function")

    func = module.selective_route_exchange_op
    _validate_signature(func)
    return func

def selective_route_exchange(
    parents: tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    rng: RandomNumberGenerator,
    *,
    custom_path: Optional[Path] = None,  # New parameter
) -> Solution:
    """
    The selective route exchange crossover (SREX) operator due to Nagata and
    Kobayashi [1]_ combines routes from both parents to generate a new
    offspring solution. It does this by carefully selecting routes from the
    second parent that could be exchanged with routes from the first parent.
    This often results in incomplete offspring that can then be repaired using
    a search method.

    .. note::

       Since SREX exchanges *routes*, it is not an appropriate crossover
       operator for TSP instances where each solution consists of just one
       route. SREX warns when used for TSPs, and another crossover operator
       should ideally be used when solving such instances.

    Parameters
    ----------
    parents
        The two parent solutions to create an offspring from.
    data
        The problem instance.
    cost_evaluator
        The cost evaluator used to evaluate the offspring.
    rng
        The random number generator to use.

    Returns
    -------
    Solution
        A new offspring.

    References
    ----------
    .. [1] Nagata, Y., & Kobayashi, S. (2010). A Memetic Algorithm for the
           Pickup and Delivery Problem with Time Windows Using Selective Route
           Exchange Crossover. *Parallel Problem Solving from Nature*, PPSN XI,
           536 - 545.
    """
    
    if data.num_vehicles == 1:
        msg = """
        SREX always returns one of the parent solutions for TSP instances. 
        Consider using a different crossover.
        """
        warn(msg, TspWarning)

    first, second = parents

    if first.num_clients() == 0:
        return second

    if second.num_clients() == 0:
        return first

    idx1 = rng.randint(first.num_routes())
    idx2 = idx1 if idx1 < second.num_routes() else 0
    max_routes_to_move = min(first.num_routes(), second.num_routes())
    num_routes_to_move = rng.randint(max_routes_to_move) + 1

    # # Attempt LLM-generated crossover first
    # if custom_path:
    #     try:
    #         custom = _load_custom_crossover(custom_path)
    #         offspring = custom(parents, data, cost_evaluator, (idx1, idx2), num_routes_to_move)
    #         # sanity check
    #         if not isinstance(offspring, Solution):
    #             raise ValueError(f"Custom returned {type(offspring)}, expected Solution")
    #         return offspring
    #     except Exception as e:
    #         print(f"Custom SREX failed: {e}")

    # # Attempt LLM-generated crossover first
    # if custom_path:
    #     try:
    #         custom = _load_custom_crossover(custom_path)
    #         return custom(parents, data, cost_evaluator, (idx1, idx2), num_routes_to_move) # (parents, data, indices)
    #     except Exception as e:
    #         print(f"Custom SREX failed: {e}")

    # Attempt LLM-generated crossover first
    if custom_path:
        custom = _load_custom_crossover(custom_path)
        return custom(parents, data, cost_evaluator, (idx1, idx2), num_routes_to_move) # (parents, data, indices)

    # print("SREXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX")
    
    # Original C++ fallback with all safety checks
    return _srex(
        parents, data, cost_evaluator, (idx1, idx2), num_routes_to_move
    )