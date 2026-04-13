from __future__ import annotations

import json
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from .schemas import Float2D, Instance, Solution
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

EPS = 1e-10


def compute_distance_matrix(coords: Float2D) -> Float2D:
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt((diff ** 2).sum(axis=-1)).astype(np.float64)


def _parse_vrp(vrp_path: Path) -> dict:
    text = vrp_path.read_text(encoding="utf-8")
    lines = text.splitlines()

    header: dict[str, str] = {}
    coord_lines: list[str] = []
    demand_lines: list[str] = []
    depot_indices: list[int] = []

    section: str | None = None
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped == "EOF":
            continue

        if stripped == "NODE_COORD_SECTION":
            section = "coord"
            continue
        elif stripped == "DEMAND_SECTION":
            section = "demand"
            continue
        elif stripped == "DEPOT_SECTION":
            section = "depot"
            continue
        elif stripped.endswith("SECTION"):
            section = "other"
            continue

        if section is None:
            match = re.match(r"^(.+?)\s*:\s*(.+)$", stripped)
            if match:
                header[match.group(1).strip().upper()] = match.group(2).strip().strip('"')
        elif section == "coord":
            coord_lines.append(stripped)
        elif section == "demand":
            demand_lines.append(stripped)
        elif section == "depot":
            val = int(re.split(r"\s+", stripped)[0])
            if val >= 0:
                depot_indices.append(val)

    coords = np.array(
        [[float(v) for v in re.split(r"\s+", ln)[1:3]] for ln in coord_lines],
        dtype=np.float64,
    )
    demand = np.array(
        [float(re.split(r"\s+", ln)[1]) for ln in demand_lines],
        dtype=np.float64,
    )
    capacity = float(header.get("CAPACITY", "0"))

    raw_dist = header.get("DISTANCE", None)
    if raw_dist is None or float(raw_dist) <= 0:
        distance_limit = float("inf")
    else:
        distance_limit = float(raw_dist)

    return {
        "name": header.get("NAME", vrp_path.stem),
        "coords": coords,
        "demand": demand,
        "capacity": capacity,
        "distance_limit": distance_limit,
    }


def _parse_sol(sol_path: Path) -> float | None:
    text = sol_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        match = re.match(r"^Cost\s+([0-9.eE+\-]+)", line.strip())
        if match:
            return float(match.group(1))
    return None


def load_instance(
    vrp_path: Path,
    bks_lookup: dict[str, float] | None = None,
) -> Instance:
    raw = _parse_vrp(vrp_path)
    coords = raw["coords"]
    demand = raw["demand"]

    name = vrp_path.stem
    n_customers = int(len(coords) - 1)
    dist = compute_distance_matrix(coords)

    if bks_lookup is None:
        bks = float("nan")
    else:
        bks = float(bks_lookup.get(name, float("nan")))

    return Instance(
        name=name,
        n_customers=n_customers,
        coords=coords,
        demand=demand,
        capacity=raw["capacity"],
        distance_limit=raw["distance_limit"],
        distance_matrix=dist,
        bks=bks,
    )


def load_bks(bks_json: Path) -> dict[str, float]:
    with open(bks_json, encoding="utf-8") as f:
        raw = json.load(f)
    return {str(k): float(v) for k, v in raw.items()}


def build_bks_from_sol_files(data_dir: Path, output: Path) -> dict[str, float]:
    bks: dict[str, float] = {}
    for sol_path in sorted(data_dir.rglob("*.sol")):
        try:
            cost = _parse_sol(sol_path)
            if cost is None:
                continue
            bks[sol_path.stem] = cost
        except Exception as exc:
            print(f"failed to parse {sol_path}: {exc}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(bks, f, indent=2, sort_keys=True)
    return bks


def route_distance(route: list[int], dist: Float2D) -> float:
    if len(route) < 2:
        return 0.0
    return float(sum(dist[route[i], route[i + 1]] for i in range(len(route) - 1)))


def solution_from_routes(
    routes: list[list[int]], instance: Instance
) -> Solution:
    dist = instance.distance_matrix
    demand = instance.demand
    total = 0.0
    loads: list[float] = []
    feasible = True

    for r in routes:
        if len(r) < 2 or r[0] != 0 or r[-1] != 0:
            feasible = False
        d = route_distance(r, dist)
        load = float(demand[np.asarray(r, dtype=np.int64)].sum())
        total += d
        loads.append(load)
        if load > instance.capacity + EPS:
            feasible = False
        if d > instance.distance_limit + EPS:
            feasible = False

    non_empty = [r for r in routes if len(r) > 2]
    return Solution(
        routes=routes,
        distance=total,
        loads=loads,
        feasible=feasible,
        n_vehicles=len(non_empty),
    )


def plot_instance(
    instance: Instance,
    ax: "Axes | None" = None,
) -> "Figure":
    """Visualise instance nodes"""

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))
    else:
        fig = ax.figure

    coords = instance.coords
    if coords.shape[0] > 1:
        sc = ax.scatter(
            coords[1:, 0],
            coords[1:, 1],
            c=instance.demand[1:],
            cmap="YlOrRd",
            s=30,
            zorder=3,
            label="klienci",
        )
        fig.colorbar(sc, ax=ax, label="zapotrzebowanie")
    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        marker="*",
        s=300,
        c="red",
        zorder=4,
        label="depot",
    )

    for i in range(1, len(coords)):
        ax.annotate(str(i), (coords[i, 0], coords[i, 1]), fontsize=6, ha="center", va="bottom")

    ax.set_title(f"{instance.name} | n={instance.n_customers} | Q={instance.capacity}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_routes(
    instance: Instance,
    solution: Solution,
    algo: str,
    ax: "Axes | None" = None,
) -> "Figure":
    """Plot one Solution on a 2D plane with each route in a distinct colour."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))
    else:
        fig = ax.figure

    coords = instance.coords
    n_routes = len(solution.routes)
    cmap = plt.get_cmap("tab10") if n_routes <= 10 else plt.get_cmap("tab20")
    cmap_n = 10 if n_routes <= 10 else 20

    for i, route in enumerate(solution.routes):
        if len(route) < 2:
            continue
        color = cmap(i % cmap_n)
        xs = coords[route, 0]
        ys = coords[route, 1]
        ax.plot(xs, ys, color=color, linewidth=1.5, zorder=2)
        clients = route[1:-1]
        if clients:
            ax.scatter(
                coords[clients, 0],
                coords[clients, 1],
                color=color,
                s=30,
                zorder=3,
            )

    ax.scatter(
        coords[0, 0],
        coords[0, 1],
        marker="s",
        s=120,
        c="darkred",
        edgecolors="black",
        linewidths=1.5,
        zorder=5,
        label="depot",
    )

    bks_part = f" | BKS={instance.bks:.1f}" if not np.isnan(instance.bks) else ""
    ax.set_title(
        f"{instance.name} | {algo} | "
        f"len={solution.distance:.1f} | "
        f"veh={solution.n_vehicles}{bks_part}"
    )
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=8)
    return fig


def plot_routes_from_json(
    instance: Instance,
    routes_json: str,
    algo: str,
    best_len: float | None = None,
    ax: "Axes | None" = None,
) -> "Figure":
    """Build a Solution from a JSON routes string and plot it."""
    routes: list[list[int]] = json.loads(routes_json)
    sol = solution_from_routes(routes, instance)
    if best_len is not None:
        sol.distance = best_len
    return plot_routes(instance, sol, algo, ax=ax)
