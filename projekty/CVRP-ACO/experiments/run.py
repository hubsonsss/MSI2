from __future__ import annotations

import argparse
import csv
import itertools
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from joblib import Parallel, delayed
from tqdm import tqdm

from src.aco_base import AntSystem
from src.aco_KM import KMeansACO
from src.aco_mmas import MMAS
from src.greedy import nearest_neighbor
from src.schemas import ACOParams, Float1D, Instance, RunResult
from src.utils import load_bks, load_instance

ALGO_REGISTRY: dict[str, type[AntSystem]] = {
    "AS": AntSystem,
    "MMAS": MMAS,
    "KMeansACO": KMeansACO,
}


def _preload_instances(
    instance_rels: list[str],
    data_dir: Path,
    bks_lookup: dict[str, float],
) -> dict[str, tuple[Instance, float]]:
    cache: dict[str, tuple[Instance, float]] = {}
    for inst_rel in instance_rels:
        if inst_rel in cache:
            continue
        instance = load_instance(data_dir / f"{inst_rel}.vrp", bks_lookup)
        ref_length = float(nearest_neighbor(instance).distance)
        cache[inst_rel] = (instance, ref_length)
    return cache


EXPERIMENT_COLUMNS = [
    "instance",
    "algorithm",
    "seed",
    "best_len",
    "delta_bks",
    "time_s",
    "best_iter",
    "n_vehicles",
    "feasible",
    "bks",
    "routes",
]

GRIDSEARCH_COLUMNS = [
    "instance",
    "algorithm",
    "seed",
    "alpha",
    "beta",
    "rho",
    "gamma",
    "best_len",
    "delta_bks",
    "time_s",
    "best_iter",
    "routes",
]


def _load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    assert isinstance(raw, dict)
    return raw


def _params_from_cfg(cfg: dict[str, Any]) -> ACOParams:
    p = cfg.get("params") or {}
    return ACOParams(
        alpha=float(p.get("alpha", 1.0)),
        beta=float(p.get("beta", 2.0)),
        rho=float(p.get("rho", 0.1)),
        Q=float(p.get("Q", 1.0)),
        m_ants=p.get("m_ants"),
        t_max=int(p.get("t_max", 200)),
        gamma=float(p.get("gamma", 3.0)),
    )


def _read_existing(csv_path: Path) -> set[tuple[str, str, int]]:
    if not csv_path.exists():
        return set()
    seen: set[tuple[str, str, int]] = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add((row["instance"], row["algorithm"], int(row["seed"])))
            except (KeyError, ValueError):
                continue
    return seen


def _run_experiment_task(
    instance: Instance,
    ref_length: float,
    algo: str,
    seed: int,
    params: ACOParams,
) -> tuple[RunResult, Float1D | None]:
    if algo == "greedy":
        t0 = time.perf_counter()
        sol = nearest_neighbor(instance)
        elapsed = time.perf_counter() - t0
        bks = instance.bks
        delta = float("nan") if np.isnan(bks) or bks <= 0 else (sol.distance - bks) / bks * 100.0
        result = RunResult(
            instance=instance.name,
            algorithm="greedy",
            seed=seed,
            best_solution=sol,
            best_iter=0,
            delta_bks=delta,
            time_s=elapsed,
            history=np.array([sol.distance], dtype=np.float64),
        )
        return result, None

    algo_cls = ALGO_REGISTRY[algo]
    rng = np.random.default_rng(seed)
    runner = algo_cls(
        instance=instance,
        params=params,
        rng=rng,
        seed=seed,
        ref_length=ref_length,
    )
    result = runner.run()
    return result, result.history


def _experiment_row(r: RunResult, bks: float) -> dict[str, Any]:
    delta = "" if np.isnan(r.delta_bks) else f"{r.delta_bks:.4f}"
    bks_str = "" if (np.isnan(bks) or bks <= 0) else f"{bks:.4f}"
    return {
        "instance": r.instance,
        "algorithm": r.algorithm,
        "seed": r.seed,
        "best_len": f"{r.best_solution.distance:.4f}",
        "delta_bks": delta,
        "time_s": f"{r.time_s:.6f}",
        "best_iter": r.best_iter,
        "n_vehicles": r.best_solution.n_vehicles,
        "feasible": int(r.best_solution.feasible),
        "bks": bks_str,
        "routes": json.dumps(r.best_solution.routes),
    }


def run_experiment(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    algorithms: list[str] = list(cfg.get("algorithms") or ["greedy", "AS", "MMAS", "KMeansACO"])
    instances: list[str] = list(cfg["instances"])
    seeds: list[int] = list(cfg.get("seeds") or list(range(1, 21)))
    params = _params_from_cfg(cfg)

    args.output.mkdir(parents=True, exist_ok=True)
    history_dir = args.output / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"{cfg.get('name', 'runs')}.csv"

    bks_lookup = load_bks(args.bks) if args.bks.exists() else {}

    tasks: list[tuple[str, str, int]] = []
    for inst_rel in instances:
        for algo in algorithms:
            if algo == "greedy":
                tasks.append((inst_rel, algo, 0))
            else:
                for s in seeds:
                    tasks.append((inst_rel, algo, s))

    existing = _read_existing(csv_path)
    pending = [t for t in tasks if (Path(t[0]).name, t[1], t[2]) not in existing]

    print(
        f"[experiment] Config: {args.config} | algorithms: {algorithms} | "
        f"instances: {len(instances)} | seeds: {len(seeds)}"
    )
    print(
        f"Total tasks: {len(tasks)} | pending: {len(pending)} | done: {len(tasks) - len(pending)}"
    )
    if not pending:
        print("Nothing to do.")
        return

    pending_instances = list({t[0] for t in pending})
    print(f"Preloading {len(pending_instances)} instance(s)...")
    inst_cache = _preload_instances(pending_instances, args.data_dir, bks_lookup)

    def _task(t: tuple[str, str, int]) -> tuple[RunResult, Float1D | None]:
        inst_rel, algo, seed = t
        instance, ref_length = inst_cache[inst_rel]
        return _run_experiment_task(instance, ref_length, algo, seed, params)

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        gen = Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_task)(t) for t in pending
        )
        with tqdm(total=len(pending), desc="runs") as pbar:
            for result, history in gen:
                bks = bks_lookup.get(result.instance, float("nan"))
                writer.writerow(_experiment_row(result, bks))
                f.flush()
                if history is not None:
                    hist_path = (
                        history_dir / f"{result.algorithm}_{result.instance}_s{result.seed}.csv"
                    )
                    with open(hist_path, "w", encoding="utf-8", newline="") as hf:
                        hwriter = csv.writer(hf)
                        hwriter.writerow(["iter", "best_so_far"])
                        for i, v in enumerate(history):
                            hwriter.writerow([i, f"{float(v):.4f}"])
                if result.route_history:
                    rh_path = (
                        history_dir
                        / f"{result.algorithm}_{result.instance}_s{result.seed}_routes.json"
                    )
                    rh_data = [
                        {"iter": t, "distance": d, "routes": r} for t, d, r in result.route_history
                    ]
                    with open(rh_path, "w", encoding="utf-8") as rhf:
                        json.dump(rh_data, rhf)
                pbar.update(1)

    print(f"Done. Results -> {csv_path}")


def _run_gridsearch_task(
    instance: Instance,
    ref_length: float,
    algo: str,
    seed: int,
    alpha: float,
    beta: float,
    rho: float,
    gamma: float,
    t_max: int,
) -> dict[str, Any]:
    params = ACOParams(
        alpha=alpha,
        beta=beta,
        rho=rho,
        Q=1.0,
        m_ants=None,
        t_max=t_max,
        gamma=gamma,
    )
    rng = np.random.default_rng(seed)
    runner = ALGO_REGISTRY[algo](
        instance=instance,
        params=params,
        rng=rng,
        seed=seed,
        ref_length=ref_length,
    )
    result = runner.run()
    delta = "" if np.isnan(result.delta_bks) else f"{result.delta_bks:.4f}"
    return {
        "instance": instance.name,
        "algorithm": algo,
        "seed": seed,
        "alpha": alpha,
        "beta": beta,
        "rho": rho,
        "gamma": gamma,
        "best_len": f"{result.best_solution.distance:.4f}",
        "delta_bks": delta,
        "time_s": f"{result.time_s:.4f}",
        "best_iter": result.best_iter,
        "routes": json.dumps(result.best_solution.routes),
    }


GridKey = tuple[str, str, int, float, float, float, float]


def _read_existing_gridsearch(csv_path: Path) -> set[GridKey]:
    if not csv_path.exists():
        return set()
    seen: set[GridKey] = set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                seen.add(
                    (
                        row["instance"],
                        row["algorithm"],
                        int(row["seed"]),
                        float(row["alpha"]),
                        float(row["beta"]),
                        float(row["rho"]),
                        float(row["gamma"]),
                    )
                )
            except (KeyError, ValueError):
                continue
    return seen


def run_gridsearch(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    algorithms: list[str] = list(cfg.get("algorithms") or ["AS", "MMAS", "KMeansACO"])
    instances: list[str] = list(cfg["instances"])
    seeds: list[int] = list(cfg.get("seeds") or [1, 2, 3, 4, 5])
    grid = cfg["grid"]
    alphas: list[float] = [float(x) for x in grid["alpha"]]
    betas: list[float] = [float(x) for x in grid["beta"]]
    rhos: list[float] = [float(x) for x in grid["rho"]]
    gammas: list[float] = [float(x) for x in grid.get("gamma", [3.0])]
    t_max = int((cfg.get("params") or {}).get("t_max", 200))

    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / f"{cfg.get('name', 'gridsearch')}.csv"
    bks_lookup = load_bks(args.bks) if args.bks.exists() else {}

    Task = tuple[str, str, int, float, float, float, float]
    tasks: list[Task] = []
    for algo in algorithms:
        gammas_iter = gammas if algo == "KMeansACO" else [3.0]
        for inst in instances:
            for a, b, r, g, s in itertools.product(alphas, betas, rhos, gammas_iter, seeds):
                tasks.append((inst, algo, s, a, b, r, g))

    existing = _read_existing_gridsearch(csv_path)
    pending = [
        t for t in tasks if (Path(t[0]).name, t[1], t[2], t[3], t[4], t[5], t[6]) not in existing
    ]

    print(f"[gridsearch] Grid tasks: {len(tasks)} | algorithms={algorithms}")
    print(
        f"Total tasks: {len(tasks)} | pending: {len(pending)} | done: {len(tasks) - len(pending)}"
    )
    if not pending:
        print("Nothing to do.")
        return

    pending_instances = list({t[0] for t in pending})
    print(f"Preloading {len(pending_instances)} instance(s)...")
    inst_cache = _preload_instances(pending_instances, args.data_dir, bks_lookup)

    def _task(t: Task) -> dict[str, Any]:
        inst, algo, seed, a, b, r, g = t
        instance, ref_length = inst_cache[inst]
        return _run_gridsearch_task(
            instance,
            ref_length,
            algo,
            seed,
            a,
            b,
            r,
            g,
            t_max,
        )

    write_header = not csv_path.exists() or csv_path.stat().st_size == 0
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=GRIDSEARCH_COLUMNS)
        if write_header:
            writer.writeheader()
            f.flush()

        gen = Parallel(n_jobs=args.n_jobs, backend="loky", return_as="generator_unordered")(
            delayed(_task)(t) for t in pending
        )
        with tqdm(total=len(pending), desc="grid") as pbar:
            for row in gen:
                writer.writerow(row)
                f.flush()
                pbar.update(1)

    print(f"Done -> {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run CVRP-ACO experiments or grid search (auto-detected from YAML config)."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--bks", type=Path, default=Path("bks.json"))
    args = parser.parse_args()

    cfg = _load_config(args.config)

    if "grid" in cfg:
        run_gridsearch(cfg, args)
    else:
        run_experiment(cfg, args)


if __name__ == "__main__":
    main()
