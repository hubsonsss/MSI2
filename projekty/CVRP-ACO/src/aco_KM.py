from __future__ import annotations

import math
from sklearn.cluster import KMeans

import numpy as np

from .aco_base import AntSystem
from .schemas import Float2D


class KMeansACO(AntSystem):
    algorithm_name = "KMeansACO"

    def init_pheromones(self) -> Float2D:
        inst = self.instance
        demand_sum = float(inst.demand.sum())
        k_c = max(1, math.ceil(demand_sum / inst.capacity))

        client_coords = inst.coords[1:]
        km = KMeans(n_clusters=k_c, random_state=self.seed, n_init=10)
        labels = km.fit_predict(client_coords)

        full_labels = np.full(self.n_nodes, -1, dtype=np.int64)
        full_labels[1:] = labels

        gamma = self.params.gamma
        tau_inner = gamma * self.tau_0
        tau_outer = self.tau_0 / gamma

        same = full_labels[:, None] == full_labels[None, :]
        tau = np.where(same, tau_inner, tau_outer).astype(np.float64)
        tau[0, :] = self.tau_0
        tau[:, 0] = self.tau_0
        return tau
