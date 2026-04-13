from __future__ import annotations

import numpy as np

from .schemas import Instance, Solution
from .utils import solution_from_routes


def nearest_neighbor(instance: Instance) -> Solution:
    """Deterministic Nearest Neighbor heuristic for CVRP."""
    n = instance.n_customers
    dist = instance.distance_matrix
    demand = instance.demand
    capacity = instance.capacity
    s_max = instance.distance_limit

    unvisited = np.ones(n + 1, dtype=bool)
    unvisited[0] = False

    routes: list[list[int]] = []
    while unvisited.any():
        route: list[int] = [0]
        current = 0
        load = 0.0
        travelled = 0.0
        stepped = False

        while True:
            candidates = np.where(unvisited)[0]
            if candidates.size == 0:
                break

            step = dist[current, candidates]
            return_leg = dist[candidates, 0]
            new_load = load + demand[candidates]
            new_travel = travelled + step + return_leg

            feasible_mask = (new_load <= capacity) & (new_travel <= s_max)
            if not feasible_mask.any():
                break

            valid = candidates[feasible_mask]
            idx = int(np.argmin(dist[current, valid]))
            nxt = int(valid[idx])

            route.append(nxt)
            unvisited[nxt] = False
            load += float(demand[nxt])
            travelled += float(dist[current, nxt])
            current = nxt
            stepped = True

        if not stepped:
            remaining = np.where(unvisited)[0].tolist()
            raise RuntimeError(
                f"NN cannot place remaining clients {remaining} "
                f"— instance likely infeasible under C={capacity}, S_max={s_max}"
            )

        route.append(0)
        routes.append(route)

    return solution_from_routes(routes, instance)
