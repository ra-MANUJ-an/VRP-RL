"""
Reward scoring module for TSP problems.
This module provides functions to evaluate solutions to TSP problems.
"""

import re
import numpy as np
import math

def parse_tour_from_answer(solution_str, method='strict'):
    """
    Extract the tour from the solution string.
    
    Args:
        solution_str: String containing the model's output
        method: 'strict' requires proper <answer> tags, 'flexible' tries to extract any sequence of numbers
        
    Returns:
        list: The extracted tour as a list of integers, or None if extraction fails
    """
    if method == 'strict':
        # Extract tour from <answer> tags
        answer_match = re.search(r"<answer>(.*?)</answer>", solution_str, re.DOTALL)
        if answer_match is None:
            return None
        
        tour_str = answer_match.group(1).strip()
    else:  # flexible
        # Just find any sequence of numbers
        tour_str = solution_str
    
    # Parse the tour (handling various formats)
    tour = []
    for item in re.split(r'[,\s→\->]+', tour_str):
        try:
            if item.strip():
                tour.append(int(item.strip()))
        except ValueError:
            continue
    
    # Return None if no numbers were found
    if not tour:
        return None
        
    return tour

def evaluate_tsp_tour(tour, dist_matrix):
    """
    Calculate the total distance of a TSP tour.
    
    Args:
        tour: List of city indices (1-indexed)
        dist_matrix: Distance matrix between cities
        
    Returns:
        float: Total tour distance
    """
    total_distance = 0
    n = len(tour)
    
    for i in range(n - 1):
        city1 = tour[i] - 1  # Adjust for 0-indexing
        city2 = tour[i + 1] - 1
        total_distance += dist_matrix[city1][city2]
    
    # Add distance from last city back to first
    total_distance += dist_matrix[tour[-1] - 1][tour[0] - 1]
    
    return total_distance

def is_valid_tour(tour, num_cities):
    """
    Check if a tour is valid (visits each city exactly once).
    
    Args:
        tour: List of city indices
        num_cities: Number of cities
        
    Returns:
        bool: True if the tour is valid
    """
    # Check if tour has correct length and contains each city exactly once
    unique_cities = set(tour)
    return len(tour) == num_cities and len(set(tour)) == num_cities and all(1 <= city <= num_cities for city in tour)

def compute_score(solution_str, ground_truth=None, distance_matrix=None, num_cities=None, 
                  method='strict', format_score=0.1, valid_tour_score=0.4, optimal_score=1.0,
                  optimal_distance=None):
    """
    Score a TSP solution.
    
    Args:
        solution_str: The solution text from the model
        ground_truth: The ground truth optimal tour (optional)
        distance_matrix: Distance matrix between cities
        num_cities: Number of cities in the problem
        method: Method to extract the solution ('strict' or 'flexible')
        format_score: Score for correctly formatting the answer
        valid_tour_score: Score for producing a valid tour
        optimal_score: Maximum score for the optimal solution
        optimal_distance: The distance of the known optimal tour (optional)
    
    Returns:
        float: Score between 0 and 1
    """
    # Convert distance matrix to numpy array if it's a list
    if isinstance(distance_matrix, list):
        distance_matrix = np.array(distance_matrix)
    
    # Extract the tour from the solution
    extracted_tour = parse_tour_from_answer(solution_str, method)
    
    # If no tour was extracted, return format score
    if extracted_tour is None:
        return 0.0
    
    # Check if it has the correct format (this is already done by parse_tour_from_answer)
    # Give format score for having a correctly formatted answer
    score = format_score
    
    # Check if it's a valid tour
    if not is_valid_tour(extracted_tour, num_cities):
        return score
    
    # Add the valid tour score
    score = format_score + valid_tour_score
    
    # Evaluate the quality of the tour if we have a distance matrix
    if distance_matrix is not None:
        # Calculate the distance of the extracted tour
        tour_distance = evaluate_tsp_tour(extracted_tour, distance_matrix)
        
        # If we have the optimal tour or distance, compare against it
        if optimal_distance is not None:
            # Calculate ratio to optimal (1.0 means optimal, higher is worse)
            ratio = tour_distance / optimal_distance
            
            # Scale remaining score (up to optimal_score - valid_tour_score - format_score)
            remaining_score = optimal_score - valid_tour_score - format_score
            
            # High score if close to optimal, scaled down as it gets worse
            if ratio <= 1.05:  # Within 5% of optimal
                quality_score = remaining_score
            else:
                # Linear decay as ratio goes from 1.05 to 2.0
                quality_score = max(0.0, remaining_score * (1 - (ratio - 1.05) / 0.95))
                
            score += quality_score
        
        # If we have the ground truth tour but not the optimal distance
        elif ground_truth is not None:
            # Calculate the optimal distance
            optimal_distance = evaluate_tsp_tour(ground_truth, distance_matrix)
            # Recursive call with the computed optimal distance
            return compute_score(
                solution_str, ground_truth, distance_matrix, num_cities, 
                method, format_score, valid_tour_score, optimal_score, optimal_distance
            )
        
        # If we don't have any reference, use a heuristic (nearest neighbor)
        else:
            # Compute nearest neighbor solution as a baseline
            nn_distance = compute_nearest_neighbor_distance(distance_matrix)
            
            # Calculate ratio to nearest neighbor (lower is better)
            ratio = tour_distance / nn_distance
            
            # Scale remaining score
            remaining_score = optimal_score - valid_tour_score - format_score
            
            # If our tour is better than NN, give it a high score
            if ratio <= 1.0:
                quality_score = remaining_score * (1.0 - ratio/2)  # Still reward improvement over NN
            else:
                # Linear decay as ratio goes from 1.0 to 2.0
                quality_score = max(0.0, remaining_score * (1 - (ratio - 1.0)))
                
            score += quality_score
    
    return score

def compute_nearest_neighbor_distance(dist_matrix):
    """
    Compute the total distance of a tour constructed using the nearest neighbor heuristic.
    
    Args:
        dist_matrix: Distance matrix between cities
        
    Returns:
        float: Total distance of the nearest neighbor tour
    """
    num_cities = len(dist_matrix)
    visited = [False] * num_cities
    current = 0  # Start from the first city
    nn_tour = [current + 1]  # 1-indexed tour
    visited[current] = True
    nn_distance = 0
    
    for _ in range(num_cities - 1):
        min_dist = float('inf')
        next_city = -1
        for j in range(num_cities):
            if not visited[j] and dist_matrix[current][j] < min_dist:
                min_dist = dist_matrix[current][j]
                next_city = j
        
        if next_city != -1:
            nn_distance += min_dist
            current = next_city
            nn_tour.append(current + 1)  # 1-indexed
            visited[current] = True
    
    # Add distance back to start
    nn_distance += dist_matrix[current][0]
    
    return nn_distance