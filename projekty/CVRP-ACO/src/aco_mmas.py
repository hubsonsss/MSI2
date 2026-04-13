from __future__ import annotations

import numpy as np

from .aco_base import AntSystem
from .schemas import ACOParams, Float2D, Instance, Solution


class MMAS(AntSystem):
    algorithm_name = "MMAS"

    def __init__(
        self,
        instance: Instance,
        params: ACOParams,
        rng: np.random.Generator,
        seed: int,
        ref_length: float | None = None,
    ) -> None:
        super().__init__(instance, params, rng, seed, ref_length=ref_length)
        self._tau_min, self._tau_max = self._tau_bounds(self._ref_length)

    def _tau_bounds(self, best_length: float) -> tuple[float, float]:
        rho = self.params.rho
        tau_max = 1.0 / (rho * best_length)
        tau_min = tau_max / (5.0 * max(self.instance.n_customers, 1))
        return tau_min, tau_max

    def init_pheromones(self) -> Float2D:
        self._tau_min, self._tau_max = self._tau_bounds(self._ref_length)
        return np.full(
            (self.n_nodes, self.n_nodes), self._tau_max, dtype=np.float64
        )

    def update_pheromones(
        self, tau: Float2D, ants: list[Solution]
    ) -> Float2D:
        rho = self.params.rho
        q = self.params.Q
        tau = (1.0 - rho) * tau

        best_iter_ant = min(ants, key=lambda s: s.distance)
        if (
            self.best_solution is not None
            and self.best_solution.distance <= best_iter_ant.distance
        ):
            best = self.best_solution
        else:
            best = best_iter_ant

        contrib = q / best.distance
        for route in best.routes:
            for a, b in zip(route[:-1], route[1:], strict=False):
                tau[a, b] += contrib
                tau[b, a] += contrib

        self._tau_min, self._tau_max = self._tau_bounds(best.distance)
        np.clip(tau, self._tau_min, self._tau_max, out=tau)
        return tau
