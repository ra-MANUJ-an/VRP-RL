```python
from pyvrp import CostEvaluator, Solution, PopulationParams

def select_survivors(
    population: list[Solution],
    cost_evaluator: CostEvaluator,
    params: PopulationParams,
) -> list[Solution]:
    # Initialize the population
    if not population:
        return []

    # Create a copy of the current population
    current_population = list(population)
    
    # Define whether to remove or keep a solution
    keep_solution = True
    
    # Select a subset of solutions with at least the specified quality thresholds
    selected_solutions = [
        s
        for s in current_population
        if s.is_feasible() and cost_evaluator.penalised_cost(s) >= params.lb_diversity and cost_evaluator.penalised_cost(s) <= params.ub_diversity
    ]
    
    if len(selected_solutions) == 0:
        return []
    
    # Repurpose the remaining solutions
    for _ in range(params.nb_close):
        if keep_solution:
            i = random.randint(0, len(selected_solutions) - 1)
            T = selected_solutions[i]
            selected_solutions[i] = current_population.pop()
            current_population.append(T)
            keep_solution = not keep_solution
    
    return selected_solutions
```

### Explanation:

1. **Initialization:**
   - `current_population`: A copy of the current population mix of feasible/infeasible solutions.
   - `selected_solutions`: A list that holds the selected solutions based on the criteria.

2. **Filtering:**
   - The `if len(selected_solutions) == 0:` condition checks if no solution satisfies the quality threshold. If it does, emptying the `selected_solutions` list and returning is reached.

3. **Selection:**
   - Using list comprehension, the `keep_solution` boolean is set to `True` to include the current solution in the subset.
   - A randomly selected solution is chosen to replace the current solution in `current_population`.
   - `repeat` is used to repurpose the remaining solutions.
   - The `keep_solution` boolean remains `True` as long as the current solution is selected.

4. **Return Result:**
   - The list of selected solutions is returned.

This implementation ensures that solutions are selected according to the quality thresholds defined by the given `PopulationParams` object, maintaining diversity while also prioritizing the best solution candidates.<|im_end|>