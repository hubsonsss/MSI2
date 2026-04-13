from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Float1D = NDArray[np.float64]
Float2D = NDArray[np.float64]
Int1D = NDArray[np.int64]


@dataclass(frozen=True)
class Instance:
    name: str
    n_customers: int
    coords: Float2D
    demand: Float1D
    capacity: float
    distance_limit: float
    distance_matrix: Float2D
    bks: float


@dataclass
class Solution:
    routes: list[list[int]]
    distance: float
    loads: list[float]
    feasible: bool
    n_vehicles: int


@dataclass
class ACOParams:
    alpha: float = 1.0
    beta: float = 2.0
    rho: float = 0.1
    Q: float = 1.0
    m_ants: int | None = None
    t_max: int = 200
    gamma: float = 3.0


@dataclass
class RunResult:
    instance: str
    algorithm: str
    seed: int
    best_solution: Solution
    best_iter: int
    delta_bks: float
    time_s: float
    history: Float1D
    route_history: list[tuple[int, float, list[list[int]]]] | None = None
