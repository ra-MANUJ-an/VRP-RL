```python
def select_parents(
    population: list[Solution],
    rng: RandomNumberGenerator,
    cost_evaluator: CostEvaluator,
    k: int = 2
) -> tuple[Solution, Solution]:
    # Elitism: Select the k best solutions
    elite_population = sorted(population, key=lambda sol: cost_evaluator.penalised_cost(sol))[:k]
    
    # Diversity maintenance: Ensure the parents are different
    while True:
        # Random exploration: Select two random solutions
        parent1_index = rng.randint(len(population))
        parent2_index = rng.randint(len(population))
        
        if parent1_index != parent2_index:
            parent1 = population[parent1_index]
            parent2 = population[parent2_index]
            # Ensure the parents are unique and not in the elite population
            if parent1 not in elite_population and parent2 not in elite_population:
                break
    
    return parent1, parent2
```<|im_end|>