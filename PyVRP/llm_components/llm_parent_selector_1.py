```python
def select_parents(
    population: list['Solution'],      # list of PyVRP Solution objects
    rng: 'RandomNumberGenerator',      # PyVRP RandomNumberGenerator
    cost_evaluator: 'CostEvaluator',   # PyVRP CostEvaluator
    k: int = 2                       # Number of parents to select (always 2 for this task)
) -> tuple['Solution', 'Solution']:
    # Select best k solutions based on cost
    best_solutions = sorted(population, key=lambda x: cost_evaluator.cost(x))[:k]
    
    # Select two parents
    parent1 = best_solutions[rng.randint(k)]
    parent2 = best_solutions[rng.randint(k)]
    
    # Ensure parents are different
    while parent1 == parent2:
        parent2 = best_solutions[rng.randint(k)]
    
    return parent1, parent2
```<|im_end|>