```python
from itertools import groupby

def selective_route_exchange_op(
    parents: tuple[Solution, Solution],
    data: ProblemData,
    cost_evaluator: CostEvaluator,
    start_indices: tuple[int, int],
    num_moved_routes: int
) -> Solution:
    # Check if both parents have at least one route
    if len(parents[0].routes) == 0 or len(parents[1].routes) == 0:
        raise ValueError("One or both parents must have at least one route")

    # Determine the start routes for crossover
    start_routes = [routes[0] for routes in parents if len(routes) - order(len(routes)) > num_moved_routes][start_indices[0]]
    end_routes = [routes[-1] for routes in parents if len(routes) - order(len(routes)) > num_moved_routes][start_indices[1]]

    # Get routes from both parents
    parent_routes = [parents[0].routes[start : start + num_moved_routes],
                      parents[1].routes[start : start + num_moved_routes]]

    # Attempt crossover between parent routes
    if any(parent_routes):
        if isinstance(parent_routes[0], (List, tuple)):
            c = random.randint(0, len(parent_routes[0]) - 1)
            c1, c2 = groupby(parent_routes[0])[c]
            son_routes = [merge(c2, c1), merge(c2, c2)]
        else:
            c = random.randint(0, len(parent_routes[0]) - 1)
            son_routes = [merge(c, c), merge(c, c)]

        # Calculate travel distances between son routes
        travel_distances = []
        for i in range(len(son_routes)):
            x1, y1 = son_routes[i].location()
            x2, y2 = son_routes[(i + 1) % len(son_routes)].location()
            travel_distance = cost_evaluator.distance(data, x1, y1, x2, y2)
            travel_distances.append(travel_distance)

        # Assign actions toson routes
        son_actions = []
        for i in range(len(son_routes)):
            son_action = lcfs(data, cost_evaluator, son_routes[i].start_time, son_routes[i].end_time)
            son_actions.append(son_action)

        # Create son solutions
        son_solutions = [Solution(data, routes=son_routes, actions=son_actions, travel_distances=travel_distances)]
        return son_solutions

    else:
        # If no crossover, return the original parents
        return parents
```

This function attempts to crossover between two parent solutions based on `num_moved_routes`. It ensures that each objective function is evaluated and always always returns the same result in case of a failure of a crossover attempt.<|im_end|>