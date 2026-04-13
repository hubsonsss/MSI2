from __future__ import annotations

import time

import numpy as np

from .greedy import nearest_neighbor
from .schemas import ACOParams, Float2D, Instance, RunResult, Solution
from .utils import solution_from_routes

EPS = 1e-10


class AntSystem:
    """Ant System base class for CVRP"""

    algorithm_name: str = "AS"

    def __init__(
        self,
        instance: Instance,
        params: ACOParams,
        rng: np.random.Generator,
        seed: int,
        ref_length: float | None = None,
    ) -> None:
        self.instance = instance
        self.params = params
        self.rng = rng
        self.seed = seed

        n = instance.n_customers
        self.n_nodes = n + 1
        self.m_ants = params.m_ants if params.m_ants is not None else n

        self.dist: Float2D = instance.distance_matrix
        eta = 1.0 / (self.dist + EPS)
        np.fill_diagonal(eta, 0.0)
        self.eta: Float2D = eta

        if ref_length is not None:
            self._ref_length = ref_length
        else:
            self._ref_length = float(nearest_neighbor(instance).distance)
        self.tau_0 = float(self.m_ants) / self._ref_length

        self._dist_to_depot = self.dist[0].copy()

        self.best_solution: Solution | None = None
        self.best_iter: int = 0
        self.route_history: list[tuple[int, float, list[list[int]]]] = []

    def init_pheromones(self) -> Float2D:
        return np.full(
            (self.n_nodes, self.n_nodes), self.tau_0, dtype=np.float64
        )

    def update_pheromones(
        self, tau: Float2D, ants: list[Solution]
    ) -> Float2D:
        rho = self.params.rho
        q = self.params.Q
        tau = (1.0 - rho) * tau
        for ant in ants:
            contrib = q / ant.distance
            for route in ant.routes:
                for a, b in zip(route[:-1], route[1:], strict=False):
                    tau[a, b] += contrib
                    tau[b, a] += contrib
        return tau

    def run(self) -> RunResult:
        t0 = time.perf_counter()
        tau = self.init_pheromones()
        t_max = self.params.t_max
        history = np.empty(t_max, dtype=np.float64)

        alpha = self.params.alpha
        beta = self.params.beta

        eta_beta = self.eta ** beta

        for t in range(t_max):
            score_matrix = (tau ** alpha) * eta_beta

            ants = [self._construct_ant(score_matrix) for _ in range(self.m_ants)]
            best_iter_ant = min(ants, key=lambda s: s.distance)
            if (
                self.best_solution is None
                or best_iter_ant.distance < self.best_solution.distance
            ):
                self.best_solution = best_iter_ant
                self.best_iter = t
                self.route_history.append(
                    (t, best_iter_ant.distance, best_iter_ant.routes)
                )

            tau = self.update_pheromones(tau, ants)
            history[t] = self.best_solution.distance

        elapsed = time.perf_counter() - t0
        assert self.best_solution is not None

        bks = self.instance.bks
        if np.isnan(bks) or bks <= 0:
            delta = float("nan")
        else:
            delta = (self.best_solution.distance - bks) / bks * 100.0

        return RunResult(
            instance=self.instance.name,
            algorithm=self.algorithm_name,
            seed=self.seed,
            best_solution=self.best_solution,
            best_iter=self.best_iter,
            delta_bks=delta,
            time_s=elapsed,
            history=history,
            route_history=self.route_history,
        )

    def _construct_ant(self, score_matrix: Float2D) -> Solution:
        n_nodes = self.n_nodes
        demand = self.instance.demand
        capacity = self.instance.capacity
        s_max = self.instance.distance_limit
        dist = self.dist
        dist_to_depot = self._dist_to_depot

        client_visited = np.zeros(n_nodes, dtype=bool)
        client_visited[0] = True

        routes: list[list[int]] = []
        while not client_visited.all():
            route: list[int] = [0]
            current = 0
            load = 0.0
            travelled = 0.0
            stepped = False

            while True:
                available = ~client_visited
                cap_ok = (load + demand) <= capacity
                reach_ok = (
                    travelled + dist[current] + dist_to_depot
                ) <= s_max
                fits = available & cap_ok & reach_ok
                fits[0] = False
                if not fits.any():
                    break

                scores_row = score_matrix[current]
                masked = np.where(fits, scores_row, 0.0)
                total = float(masked.sum())
                if total <= 0.0 or not np.isfinite(total):
                    idx = np.where(fits)[0]
                    nxt = int(self.rng.choice(idx))
                else:
                    probs = masked / total
                    nxt = int(self.rng.choice(n_nodes, p=probs))

                route.append(nxt)
                client_visited[nxt] = True
                load += float(demand[nxt])
                travelled += float(dist[current, nxt])
                current = nxt
                stepped = True

            if not stepped:
                remaining = np.where(~client_visited)[0].tolist()
                raise RuntimeError(
                    f"Ant cannot place remaining clients {remaining} "
                    f"— instance likely infeasible"
                )

            route.append(0)
            routes.append(route)

        return solution_from_routes(routes, self.instance)
