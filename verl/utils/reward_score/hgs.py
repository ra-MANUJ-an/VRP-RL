# hgs_reward.py
import signal
import time
from typing import Dict, Any
from pathlib import Path
from pyvrp import ProblemData, SolveParams, solve
from pyvrp.Population import Population, PopulationParams
from pyvrp.stop import MaxIterations
from pyvrp import Model, read

# Register SIGFPE handler to convert to Python exception
# This ensures floating-point errors in C/C++ extensions are caught in Python
def _handle_sigfpe(signum, frame):
    raise FloatingPointError("Floating point exception (SIGFPE) caught")

signal.signal(signal.SIGFPE, _handle_sigfpe)

def evaluate(
        solution_str: str,
        instance_path,
        baseline_cost
    ) -> Dict[str, Any]:
    """Evaluate generated code using PyVRP's built-in validation"""

    relative_path = instance_path.split("tsp_100_instances/", 1)[-1]
    new_path = f"/opt/tiger/verl_code_ai_utils/tsp_100_instances/{relative_path}"
    instance = read(new_path)
    baseline_cost = baseline_cost
    selector_path = Path("/opt/tiger/verl_code_ai_utils/VRP-RL/PyVRP/llm_components/llm_parent_selector.py")

    try:
        selector_path.write_text(solution_str)

        # params = SolveParams(
        #     population=PopulationParams(min_pop_size=25),
        #     custom_survivor_path=selector_path
        # )
        params = SolveParams(
            population=PopulationParams(min_pop_size=25),
            custom_selector_path=selector_path
        )
        llm_result = solve(
            instance,
            stop=MaxIterations(1000),
            params=params,
            display=False
        )
        improvement = baseline_cost - llm_result.cost()
        print(f"Improvement: {improvement}")
        return {
            "valid": True,
            "cost": llm_result.cost(),
            "improvement": improvement
        }
    except FloatingPointError as e:
        # Catch SIGFPE and treat as invalid evaluation
        return {"valid": False, "error": f"SIGFPE error: {e}"}
    except Exception as e:
        return {"valid": False, "error": str(e)}


def compute_score(
        solution_str,
        instance_path,
        baseline_cost
    ) -> float:
    """Reward is:
    - 1.0 if improvement >= 0
    - 0.0 if improvement < 0
    - -1.0 if code fails to run or a SIGFPE occurs
    """
    metrics = evaluate(solution_str, instance_path, baseline_cost)
    if not metrics.get("valid", False):
        return -1.0
    return 1.0 if metrics["improvement"] >= 0 else 0.0



# # hgs_reward.py
# import time
# from typing import Dict, Any
# from pathlib import Path
# from pyvrp import ProblemData, SolveParams, solve
# from pyvrp.Population import Population, PopulationParams
# from pyvrp.stop import MaxIterations
# from pyvrp import Model, read

# def evaluate(
#         solution_str: str,
#         instance_path,
#         baseline_cost
#     ) -> Dict[str, Any]:
#     """Evaluate generated code using PyVRP's built-in validation"""
#     try:
#         instance = read(instance_path)
#         # selector_path = Path("/common/home/users/m/manujm/PyVRP/llm_components/llm_selective_route_exchange_1.py")
#         selector_path = Path("/common/home/users/m/manujm/PyVRP/llm_components/llm_parent_selector.py")
#         selector_path.write_text(solution_str)

#         # params = SolveParams(
#         #     population=PopulationParams(min_pop_size=25),
#         #     custom_crossover_path=selector_path
#         # )
#         params = SolveParams(
#             population=PopulationParams(min_pop_size=25),
#             custom_selector_path=selector_path
#         )
#         llm_result = solve(
#             instance,
#             stop=MaxIterations(1000),
#             params=params,
#             display=False
#         )
#         improvement = baseline_cost - llm_result.cost()
#         return {
#             "valid": True,
#             "cost": llm_result.cost(),
#             "improvement": improvement
#         }
        
#     except Exception as e:
#         return {"valid": False, "error": str(e)}

# def compute_score(
#         solution_str,
#         instance_path,
#         baseline_cost
#     ) -> float:
#     """Reward is:
#     - 1.0 if improvement > 0
#     - 0.0 if improvement <= 0
#     - -1.0 if code fails to run
#     """
#     metrics = evaluate(solution_str, instance_path, baseline_cost)

#     # instance = read(instance_path)
#     # default_result = solve(
#     #     instance,
#     #     stop=MaxIterations(100),
#     #     display=False
#     # )
#     # print(f"Default selector best cost: {default_result.cost()}")
    
#     if not metrics["valid"]:
#         return -1.0
    
#     return 1.0 if metrics["improvement"] >= 0 else 0.0




# # hgs_reward.py
# import time
# from typing import Dict, Any
# from pathlib import Path
# from pyvrp import ProblemData, SolveParams, solve
# from pyvrp.Population import Population, PopulationParams
# from pyvrp.stop import MaxIterations
# from pyvrp import Model, read


# def evaluate(
#         solution_str: str,
#         instance_path,
#         baseline_cost
#     ) -> Dict[str, Any]:
#     """Evaluate generated code using PyVRP's built-in validation"""
#     try:
#         instance = read(instance_path)
#         selector_path = Path("/common/home/users/m/manujm/PyVRP/llm_components/llm_parent_selector.py")
#         selector_path.write_text(solution_str)
#         # Configure solver with LLM integration
#         params = SolveParams(
#             population=PopulationParams(min_pop_size=25),  # Smaller population for testing
#             custom_selector_path=selector_path
#         )
#         llm_result = solve(
#             instance,
#             stop=MaxIterations(1000),  # Short run for testing
#             params=params,
#             display=False
#         )
#         improvement = baseline_cost - llm_result.cost()
#         return {
#             "valid": True,
#             "cost": llm_result.cost(),
#             "improvement": improvement,
#             "reward": improvement / baseline_cost
#         }
        
#     except Exception as e:
#         return {"valid": False, "error": str(e)}

# def compute_score(
#         solution_str,
#         instance_path,
#         baseline_cost
#     ) -> float:
#     """Simplified reward function using Population's validation"""

#     metrics = evaluate(solution_str, instance_path, baseline_cost)
    
#     if not metrics["valid"]:
#         return -1.0  # Format/reliability penalty from Population's validation
    
#     # Combined rewards (quality + 0.2 format bonus)
#     return min(max(metrics["reward"] + 0.2, -1.0), 1.0)