import re
import os
import datasets
import numpy as np
import math
from verl.utils.hdfs_io import copy, makedirs
import argparse

def parse_portgen_file(file_path):
    """Parse a PORTGEN file and extract coordinates."""
    coords = []
    reading_coords = False
    
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line == "NODE_COORD_SECTION":
                reading_coords = True
                continue
            elif line.startswith("EOF") or len(line) == 0:
                break
            
            if reading_coords:
                parts = line.split()
                if len(parts) >= 3:
                    # Format is: node_id x_coord y_coord
                    try:
                        node_id = int(parts[0])
                        x = float(parts[1])
                        y = float(parts[2])
                        coords.append((node_id, x, y))
                    except ValueError:
                        continue
    
    return coords

def parse_solution_file(file_path):
    """Parse a solution file and extract the optimal tour."""
    tour = []
    
    with open(file_path, 'r') as f:
        lines = f.readlines()
        # Skip the first line (dimension)
        for i in range(1, len(lines)):
            line = lines[i].strip()
            if not line:
                continue
            
            # Parse the numbers in the line
            for num in line.split():
                if num.isdigit():
                    # Adding 1 because our indices start from 1, not 0
                    tour.append(int(num) + 1)
    
    return tour

def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points."""
    return math.sqrt((p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)

def create_distance_matrix(coords):
    """Create a distance matrix from coordinates."""
    n = len(coords)
    dist_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                dist_matrix[i][j] = calculate_distance(coords[i], coords[j])
    
    return dist_matrix

def format_coords_for_prompt(coords):
    """Format coordinates for the prompt."""
    formatted = "["
    for i, (node_id, x, y) in enumerate(coords):
        formatted += f"({x:.2f}, {y:.2f})"
        if i < len(coords) - 1:
            formatted += ", "
    formatted += "]"
    return formatted

def load_portgen_dataset(data_dir, solutions_dir=None):
    """
    Load TSP instances from PORTGEN dataset format and process into a structured dataset.
    
    Args:
        data_dir (str): Directory containing TSP problem files (.tsp)
        solutions_dir (str, optional): Directory containing solution files (.sol)
        
    Returns:
        datasets.Dataset: A Hugging Face dataset with processed TSP instances
    """
    all_instances = []
    
    # Walk through the directory to find data files
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.tsplib'):
                file_path = os.path.join(root, file)
                instance_name = os.path.splitext(file)[0]
                
                coords = parse_portgen_file(file_path)
                if len(coords) == 0:
                    continue
                    
                dist_matrix = create_distance_matrix(coords)
                optimal_tour = None
                
                # If solutions directory is provided, try to load the solution
                if solutions_dir:
                    solution_file = os.path.join(solutions_dir, f"{instance_name}.txt")
                    if os.path.exists(solution_file):
                        optimal_tour = parse_solution_file(solution_file)
                
                # Format coordinates for the prompt
                coords_str = format_coords_for_prompt(coords)

                # Example TSP problem with coordinates and solution
                example_coords = [(105248.00, 970395.00), (836980.00, 300250.00), (218215.00, 896502.00), 
                                  (425618.00, 864866.00), (793754.00, 443520.00), (506758.00, 896530.00), 
                                  (55635.00, 602923.00), (874446.00, 559828.00), (760620.00, 366773.00), 
                                  (401680.00, 205876.00), (636262.00, 87967.00), (62639.00, 794461.00), 
                                  (881378.00, 823038.00), (979324.00, 171577.00), (482231.00, 345039.00), 
                                  (787947.00, 995643.00), (895598.00, 974329.00), (701754.00, 343991.00), 
                                  (328365.00, 935659.00), (379544.00, 828132.00)]
                example_coords_str = format_coords_for_prompt([(i+1, x, y) for i, (x, y) in enumerate(example_coords)])
                example_optimal_tour = [1, 3, 19, 20, 4, 6, 16, 17, 13, 8, 5, 9, 18, 2, 14, 11, 10, 15, 7, 12]
                
                example_response = """<think>
I need to find the shortest tour through all 20 cities.

The optimal tour will be determined after analyzing the city coordinates and their relative positions. I've determined following as the optimal tour:
1 → 3 → 19 → 20 → 4 → 6 → 16 → 17 → 13 → 8 → 5 → 9 → 18 → 2 → 14 → 11 → 10 → 15 → 7 → 12

This tour minimizes the total distance while visiting each city exactly once.
</think>

<answer>1, 3, 19, 20, 4, 6, 16, 17, 13, 8, 5, 9, 18, 2, 14, 11, 10, 15, 7, 12</answer>"""
                
                # Create instance entry
                instance_data = {
                    "data_source": "tsp20",
                    "prompt": [{
                        "role": "system",
                        "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer."
                      },
                      {
                        "role": "user",
                        "content": f"Find the shortest tour that visits each city exactly once. Our objective is to solve Travelling salesman problem. The coordinates of the cities are: {example_coords_str}.\nShow your work in <think> </think> tags in brief. Return only the final tour as a sequence of city indices (starting from 1) in <answer> </answer> tags."
                      },
                      {
                        "role": "assistant",
                        "content": f"Let me find the solution for this TSP problem.\n{example_response}"
                      },
                      {
                        "role": "user",
                        "content": f"Find the shortest tour that visits each city exactly once. Our objective is to solve Travelling salesman problem. The coordinates of the cities are: {coords_str}.\nShow your work in <think> </think> tags in brief. Return only the final tour as a sequence of city indices (starting from 1) in <answer> </answer> tags."
                      },
                      {
                        "role": "assistant",
                        "content": "Let me find the solution for this TSP problem.\n<think>"
                    }],
                    "ability": "problem_solving",
                    "reward_model": {
                        "style": "tsp",
                        "ground_truth": optimal_tour
                    },
                    "extra_info": {
                        "name": instance_name,
                        "num_cities": len(coords),
                        "coordinates": coords,
                        "distance_matrix": dist_matrix.tolist(),
                    }
                }
                
                all_instances.append(instance_data)
    
    # Create a Hugging Face dataset
    dataset = Dataset.from_list(all_instances)
    
    return dataset

def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='/common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/tsp_20_instances', help='Directory containing TSP problem files')
    parser.add_argument('--solutions_dir', default='/common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/tsp_20_solutions', help='Directory containing solution files')
    parser.add_argument('--output_dir', default='/common/home/users/m/manujm/neuralcombinatorialoptimization/llm_finetuning/processed_tsp_20_data', help='Directory to save processed data')
    parser.add_argument('--hdfs_dir', default=None, help='HDFS directory to copy data to (optional)')
    
    args = parser.parse_args()
    
    # Load and process the dataset
    dataset = load_portgen_dataset(args.data_dir, args.solutions_dir)
    
    # Split into train/test
    # train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    # train_dataset = train_test_split["train"]
    # test_dataset = train_test_split["test"]
    
    # First split: train (90%) and test (10%)
    train_test_split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = train_test_split["train"]
    test_dataset = train_test_split["test"]
    
    # Further split test_dataset: 99% for evaluation, 1% reserved for inference.
    inference_split = test_dataset.train_test_split(test_size=0.01, seed=42)
    eval_dataset = inference_split["train"]
    inference_dataset = inference_split["test"]
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save datasets
    train_dataset.to_parquet(os.path.join(args.output_dir, 'train.parquet'))
    eval_dataset.to_parquet(os.path.join(args.output_dir, 'test.parquet'))
    inference_dataset.to_parquet(os.path.join(args.output_dir, 'inference.parquet'))
    
    print(f"Saved processed datasets to {args.output_dir}")
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(eval_dataset)}")
    print(f"Inference dataset size: {len(inference_dataset)}")
    
    # Copy to HDFS if specified
    if args.hdfs_dir is not None:
        try:
            from verl.utils.hdfs_io import copy, makedirs
            makedirs(args.hdfs_dir)
            copy(src=args.output_dir, dst=args.hdfs_dir)
            print(f"Copied data to HDFS: {args.hdfs_dir}")
        except ImportError:
            print("HDFS utilities not available. Data was not copied to HDFS.")

if __name__ == "__main__":
    import argparse
    from datasets import Dataset
    main()