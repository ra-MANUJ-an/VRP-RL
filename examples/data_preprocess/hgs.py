# File: preprocess_vrp_data.py
import os
import re
import json
import argparse
from pathlib import Path
import pandas as pd
from pyvrp import read, solve
from pyvrp.stop import MaxIterations
import time

def create_prompt() -> dict:
    # Optimal prompt structure
    PROMPT = """[INST]
Only write a Python function `select_parents` for a genetic algorithm (Hybrid Genetic Search) solving the Vehicle Routing Problem (VRP).
The function should select two parent solutions from a population to be used in crossover, aiming to produce high-quality offspring (lower cost solutions).
Provide only the code without explanations. The function will be used within the PyVRP library.
Follow these requirements:

1. Function signature:
def select_parents(
    population: list[Solution],
    rng: RandomNumberGenerator,
    cost_evaluator: CostEvaluator,
    k: int = 2
) -> tuple[Solution, Solution]:

2. Use PyVRP's RandomNumberGenerator methods:
- rng.randint(n) returns integer in [0, n)

3. Implement a strategy combining:
- Elitism (select best solutions)
- Diversity maintenance
- Random exploration

4. Prohibited:
- Import statements
- External dependencies

Class definitions used are provided below. Please use following class definitions only:

class CostEvaluator:
    def __init__(
        self,
        load_penalties: list[float],
        tw_penalty: float,
        dist_penalty: float,
    ) -> None: ...
    def load_penalty(
        self, load: int, capacity: int, dimension: int
    ) -> int: ...
    def tw_penalty(self, time_warp: int) -> int: ...
    def dist_penalty(self, distance: int, max_distance: int) -> int: ...
    def penalised_cost(self, solution: Solution) -> int: ...
    def cost(self, solution: Solution) -> int: ...

class RandomNumberGenerator:
    @overload
    def __init__(self, seed: int) -> None: ...
    @overload
    def __init__(self, state: list[int]) -> None: ...
    @staticmethod
    def max() -> int: ...
    @staticmethod
    def min() -> int: ...
    def rand(self) -> float: ...
    def randint(self, high: int) -> int: ...
    def __call__(self) -> int: ...
    def state(self) -> list[int]: ...

[/INST]

Function signature is as follows:
def select_parents(
    population: list[Solution],      # list of PyVRP Solution objects
    rng: RandomNumberGenerator,      # PyVRP RandomNumberGenerator
    cost_evaluator: CostEvaluator,   # PyVRP CostEvaluator
    k: int = 2                       # Number of parents to select (always 2 for this task)
) -> tuple[Solution, Solution]:

Here's the complete implementation:
"""
    
    """Generates the prompt structure for VRP instance"""
    return {
        "role": "system",
        "content": PROMPT
    }
#     return {
#         "role": "system",
#         "content": f"""Write a Python function called 'select_parents' for:
# - Clients: {instance_data['num_clients']}
# - Vehicles: {instance_data['num_vehicles']}
# - Baseline Cost: {instance_data['baseline_cost']}

# Requirements:
# 1. Use PyVRP's RandomNumberGenerator
# 2. Return tuple[Solution, Solution]
# 3. Optimize for cost reduction"""
#     }

def process_instance(instance_path: Path, idx: int) -> dict:
    """Processes a single VRP instance"""
    data = read(instance_path)
    result = solve(data, MaxIterations(1000))  # Get baseline solution
    
    return {
        "data_source": "vrp_instances",
        "prompt": [create_prompt()],
        "ability": "code_optimization",
        "reward_model": {
            "style": "custom",
            "ground_truth": {
                "instance_path": str(instance_path),
                "baseline_cost": result.cost(),
            }
        },
        "extra_info": {
            "split": "train",
            "index": idx,
            "instance_metadata": {
                "clients": data.num_clients,
                "vehicles": data.num_vehicles
            }
        }
    }

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', default='/common/home/users/m/manujm/neuralcombinatorialoptimization/rl_llm_metaheuristics/tsp_100_instances/100_1', help='Directory containing .tsplib files')
    parser.add_argument('--local_dir', default='/common/home/users/m/manujm/neuralcombinatorialoptimization/rl_llm_metaheuristics/processed_tsp_100_data', help='Local output directory')
    # parser.add_argument('--hdfs_dir', default=None, help='HDFS output directory (optional)')

    args = parser.parse_args()

    # Get all VRP instances
    instance_paths = list(Path(args.input_dir).glob("*.tsplib"))
    start = time.time()
    processed_data = [process_instance(path, idx) for idx, path in enumerate(instance_paths[:10000])]
    print(f"Time: {time.time() - start:.2f}s\n")

    # Split into train/test (80/20)
    split_idx = int(0.8 * len(processed_data))
    train_data = processed_data[:split_idx]
    test_data = processed_data[split_idx:]

    # Save datasets
    local_dir = Path(args.local_dir).expanduser()
    local_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train_data).to_parquet(local_dir / 'train.parquet')
    pd.DataFrame(test_data).to_parquet(local_dir / 'test.parquet')

    # # Optional HDFS upload
    # if args.hdfs_dir:
    #     from verl.utils.hdfs_io import copy, makedirs
    #     makedirs(args.hdfs_dir)
    #     copy(src=str(local_dir), dst=args.hdfs_dir)