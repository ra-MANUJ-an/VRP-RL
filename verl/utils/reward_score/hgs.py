# hgs_reward.py
import time
from typing import Dict, Any
from pathlib import Path
from pyvrp import ProblemData, SolveParams, solve
from pyvrp.Population import Population, PopulationParams
from pyvrp.stop import MaxIterations
from pyvrp import Model, read

class HGSEvaluator:
    def __init__(self, data_path: str, baseline_cost: int):

        # Load test instance (replace with your VRP file)
        self.data = read(data_path)
        self.baseline_cost = baseline_cost
        
    def evaluate(self, code: str) -> Dict[str, Any]:
        """Evaluate generated code using PyVRP's built-in validation"""
        try:
            selector_path = Path("/common/home/users/m/manujm/PyVRP/llm_components/llm_parent_selector.py")
            selector_path.write_text(code)
            # Configure solver with LLM integration
            params = SolveParams(
                population=PopulationParams(min_pop_size=25),  # Smaller population for testing
                custom_selector_path=selector_path
            )
            llm_result = solve(
                self.data,
                stop=MaxIterations(1000),  # Short run for testing
                params=params,
                display=False
            )
            improvement = self.baseline_cost - llm_result.cost()
            return {
                "valid": True,
                "cost": llm_result.cost(),
                "improvement": improvement,
                "reward": improvement / self.baseline_cost
            }
            
        except Exception as e:
            return {"valid": False, "error": str(e)}

def compute_score(solution_str: str, instance_path: str, baseline_cost: int) -> float:
    """Simplified reward function using Population's validation"""
    evaluator = HGSEvaluator(instance_path, baseline_cost)
    metrics = evaluator.evaluate(solution_str)
    
    if not metrics["valid"]:
        return -1.0  # Format/reliability penalty from Population's validation
    
    # Combined rewards (quality + 0.2 format bonus)
    return min(max(metrics["reward"] + 0.2, -1.0), 1.0)