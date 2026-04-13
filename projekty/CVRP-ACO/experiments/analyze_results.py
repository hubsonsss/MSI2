from __future__ import annotations

import argparse
import json as _json
import re
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.animation import FuncAnimation

from src.utils import load_bks, load_instance, plot_routes_from_json

ALGO_ORDER = ["greedy", "AS", "MMAS", "KMeansACO"]

SERIES_MAP = {
    "CMT1": "CMT",
    "CMT2": "CMT",
    "CMT3": "CMT",
    "CMT4": "CMT",
    "CMT5": "CMT",
    "Golden_1": "Golden",
    "Golden_2": "Golden",
    "Golden_3": "Golden",
    "Golden_4": "Golden",
    "Golden_5": "Golden",
    "X-n101-k25": "Uchoa",
    "X-n200-k36": "Uchoa",
    "X-n303-k21": "Uchoa",
    "X-n384-k52": "Uchoa",
    "X-n459-k26": "Uchoa",
}


def _series_from_name(name: str) -> str:
    if name in SERIES_MAP:
        return SERIES_MAP[name]
    if name.startswith("CMT"):
        return "CMT"
    if name.startswith("Golden"):
        return "Golden"
    if name.startswith("X-n"):
        return "Uchoa"
    return ""


CMT_N = {"CMT1": 50, "CMT2": 75, "CMT3": 100, "CMT4": 150, "CMT5": 199}
GOLDEN_N = {
    "Golden_1": 240,
    "Golden_2": 320,
    "Golden_3": 400,
    "Golden_4": 480,
    "Golden_5": 200,
}


def _n_from_name(name: str) -> int:
    if name in CMT_N:
        return CMT_N[name]
    if name in GOLDEN_N:
        return GOLDEN_N[name]
    m = re.match(r"X-n(\d+)", name)
    if m:
        return int(m.group(1))
    return 0


def load_runs(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df["series"] = df["instance"].map(_series_from_name).replace("", "other")
    df["delta_bks"] = pd.to_numeric(df["delta_bks"], errors="coerce")
    df["time_s"] = pd.to_numeric(df["time_s"], errors="coerce")
    df["best_iter"] = pd.to_numeric(df["best_iter"], errors="coerce")
    return df


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["series", "instance", "algorithm"], as_index=False).agg(
        mean_delta=("delta_bks", "mean"),
        std_delta=("delta_bks", "std"),
        min_delta=("delta_bks", "min"),
        mean_time=("time_s", "mean"),
        mean_best_iter=("best_iter", "mean"),
    )
    return grouped


def plot_boxplots(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aco_df = df[df["algorithm"].isin(ALGO_ORDER)]
    for series in aco_df["series"].unique():
        sub = aco_df[aco_df["series"] == series]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(11, 6))
        hue_order = [a for a in ALGO_ORDER if a in sub["algorithm"].unique()]
        sns.boxplot(
            data=sub,
            x="instance",
            y="delta_bks",
            hue="algorithm",
            order=sorted(sub["instance"].unique()),
            hue_order=hue_order,
            ax=ax,
        )
        ax.set_title(f"Odchylenie od BKS: seria {series}")
        ax.set_ylabel("δ (%)")
        ax.set_xlabel("Instancja")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"boxplot_{series}.png", dpi=150)
        plt.close(fig)


def plot_convergence(df: pd.DataFrame, history_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    aco_algos = ["AS", "MMAS", "KMeansACO"]
    for instance in sorted(df["instance"].unique()):
        fig, ax = plt.subplots(figsize=(9, 5))
        any_plotted = False
        for algo in aco_algos:
            seeds_for = df[(df["instance"] == instance) & (df["algorithm"] == algo)][
                "seed"
            ].unique()
            histories = []
            for s in seeds_for:
                hp = history_dir / f"{algo}_{instance}_s{int(s)}.csv"
                if not hp.exists():
                    continue
                h = pd.read_csv(hp)
                histories.append(h["best_so_far"].to_numpy())
            if not histories:
                continue
            length = min(len(h) for h in histories)
            mat = np.stack([h[:length] for h in histories])
            mean = mat.mean(axis=0)
            std = mat.std(axis=0)
            xs = np.arange(length)
            ax.plot(xs, mean, label=algo)
            ax.fill_between(xs, mean - std, mean + std, alpha=0.2)
            any_plotted = True

        if not any_plotted:
            plt.close(fig)
            continue
        ax.set_title(f"Zbieżność best-so-far: {instance}")
        ax.set_xlabel("Iteracja")
        ax.set_ylabel("Długość trasy")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out_dir / f"convergence_{instance}.png", dpi=150)
        plt.close(fig)


def plot_delta_vs_n(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df2 = df.copy()
    df2["n"] = df2["instance"].map(_n_from_name)
    df2 = df2[df2["n"] > 0]
    agg = df2.groupby(["algorithm", "instance", "n"], as_index=False)["delta_bks"].mean()

    fig, ax = plt.subplots(figsize=(9, 6))
    for algo in ALGO_ORDER:
        sub = agg[agg["algorithm"] == algo].sort_values("n")
        if sub.empty:
            continue
        ax.plot(sub["n"], sub["delta_bks"], "o-", label=algo)
    ax.set_xlabel("n (liczba klientów)")
    ax.set_ylabel("δ BKS (%)")
    ax.set_title("Odchylenie od BKS w zależności od liczby klientów")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "delta_vs_n.png", dpi=150)
    plt.close(fig)


def plot_gridsearch_heatmaps(gs_csv: Path, out_dir: Path) -> None:
    if not gs_csv.exists():
        print(f"[skip] grid search CSV not found at {gs_csv}")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(gs_csv)
    df["delta_bks"] = pd.to_numeric(df["delta_bks"], errors="coerce")
    for algo in df["algorithm"].unique():
        sub = df[df["algorithm"] == algo]
        if sub.empty:
            continue
        for rho in sorted(sub["rho"].unique()):
            sub_rho = sub[sub["rho"] == rho]
            agg = sub_rho.groupby(["alpha", "beta"], as_index=False)["delta_bks"].mean()
            pivot = agg.pivot(index="alpha", columns="beta", values="delta_bks")
            fig, ax = plt.subplots(figsize=(6, 5))
            sns.heatmap(pivot, annot=True, fmt=".2f", cmap="viridis_r", ax=ax)
            ax.set_title(f"{algo} | ρ={rho} | średnie δ BKS (%)")
            ax.set_ylabel("α (waga feromonu)")
            ax.set_xlabel("β (waga heurystyki)")
            fig.tight_layout()
            fig.savefig(out_dir / f"heatmap_{algo}_rho{rho}.png", dpi=150)
            plt.close(fig)

    # --- zbiorczy wykres ---
    algos = sorted(df["algorithm"].unique())
    rhos = sorted(df["rho"].unique())
    n_rows = len(algos)
    n_cols = len(rhos)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    for r, algo in enumerate(algos):
        sub = df[df["algorithm"] == algo]
        for c, rho in enumerate(rhos):
            ax = axes[r][c]
            sub_rho = sub[sub["rho"] == rho]
            agg = sub_rho.groupby(["alpha", "beta"], as_index=False)["delta_bks"].mean()
            pivot = agg.pivot(index="alpha", columns="beta", values="delta_bks")
            sns.heatmap(
                pivot,
                annot=True,
                fmt=".2f",
                cmap="viridis_r",
                ax=ax,
                cbar=c == n_cols - 1,
            )
            ax.set_title(f"{algo} | ρ={rho}")
            ax.set_ylabel("α" if c == 0 else "")
            ax.set_xlabel("β" if r == n_rows - 1 else "")
            if c > 0:
                ax.set_yticklabels([])

    fig.suptitle("Średnie δ BKS (%) — grid search (α, β) per algorytm i ρ", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_dir / "heatmap_all.png", dpi=150)
    plt.close(fig)


def plot_best_routes(df: pd.DataFrame, data_dir: Path, bks_path: Path, out_dir: Path) -> None:
    """For each (instance, algorithm), plot the route with lowest best_len."""
    if "routes" not in df.columns:
        print("[skip] No 'routes' column in CSV — skipping route plots.")
        return

    routes_dir = out_dir / "routes"
    routes_dir.mkdir(parents=True, exist_ok=True)

    bks_lookup = load_bks(bks_path) if bks_path.exists() else {}

    has_routes = df[df["routes"].notna() & (df["routes"] != "")]
    if has_routes.empty:
        print("[skip] No route data found — skipping route plots.")
        return

    inst_cache: dict[str, Any] = {}

    for (inst_name, algo), group in has_routes.groupby(["instance", "algorithm"]):
        best_row = group.loc[group["best_len"].idxmin()]

        series = _series_from_name(str(inst_name))
        inst_rel = f"{series}/{inst_name}" if series else str(inst_name)

        if inst_rel not in inst_cache:
            vrp_path = data_dir / f"{inst_rel}.vrp"
            if not vrp_path.exists():
                continue
            inst_cache[inst_rel] = load_instance(vrp_path, bks_lookup)

        instance = inst_cache[inst_rel]
        fig = plot_routes_from_json(
            instance,
            str(best_row["routes"]),
            algo=str(algo),
            best_len=float(best_row["best_len"]),
        )
        fname = f"route_{inst_name}_{algo}.png"
        fig.savefig(routes_dir / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Route plots -> {routes_dir}")


def plot_route_animation(
    df: pd.DataFrame,
    history_dir: Path,
    data_dir: Path,
    bks_path: Path,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    anim_dir = out_dir / "animations"
    anim_dir.mkdir(parents=True, exist_ok=True)

    bks_lookup = load_bks(bks_path) if bks_path.exists() else {}
    aco_algos = ["AS", "MMAS", "KMeansACO"]
    inst_cache: dict[str, Any] = {}

    for instance_name in sorted(df["instance"].unique()):
        for algo in aco_algos:
            seeds = df[(df["instance"] == instance_name) & (df["algorithm"] == algo)][
                "seed"
            ].unique()
            if len(seeds) == 0:
                continue

            rh_path = history_dir / f"{algo}_{instance_name}_s{int(seeds[0])}_routes.json"
            if not rh_path.exists():
                continue

            with open(rh_path, encoding="utf-8") as f:
                snapshots = _json.load(f)
            if not snapshots:
                continue

            series = _series_from_name(instance_name)
            inst_rel = f"{series}/{instance_name}" if series else instance_name
            if inst_rel not in inst_cache:
                vrp_path = data_dir / f"{inst_rel}.vrp"
                if not vrp_path.exists():
                    continue
                inst_cache[inst_rel] = load_instance(vrp_path, bks_lookup)
            instance = inst_cache[inst_rel]

            coords = instance.coords
            fig, ax = plt.subplots(figsize=(8, 8))

            bks_part = f" | BKS={instance.bks:.1f}" if not np.isnan(instance.bks) else ""

            def _draw(
                frame_idx: int,
                *,
                _ax: Any = ax,
                _snapshots: list[Any] = snapshots,
                _coords: Any = coords,
                _instance_name: str = instance_name,
                _algo: str = algo,
                _bks_part: str = bks_part,
            ) -> None:
                _ax.clear()
                snap = _snapshots[frame_idx]
                routes = snap["routes"]
                dist = snap["distance"]
                iteration = snap["iter"]

                cmap = plt.get_cmap("tab10") if len(routes) <= 10 else plt.get_cmap("tab20")
                cmap_n = 10 if len(routes) <= 10 else 20

                for i, route in enumerate(routes):
                    if len(route) < 2:
                        continue
                    color = cmap(i % cmap_n)
                    xs = _coords[route, 0]
                    ys = _coords[route, 1]
                    _ax.plot(xs, ys, color=color, linewidth=1.5, zorder=2)
                    clients = route[1:-1]
                    if clients:
                        _ax.scatter(
                            _coords[clients, 0],
                            _coords[clients, 1],
                            color=color,
                            s=30,
                            zorder=3,
                        )

                _ax.scatter(
                    _coords[0, 0],
                    _coords[0, 1],
                    marker="s",
                    s=120,
                    c="darkred",
                    edgecolors="black",
                    linewidths=1.5,
                    zorder=5,
                )
                _ax.set_title(
                    f"{_instance_name} | {_algo} | iter={iteration} | len={dist:.1f}{_bks_part}"
                )
                _ax.set_aspect("equal", adjustable="datalim")
                _ax.grid(True, alpha=0.3)

            anim = FuncAnimation(
                fig,
                _draw,  # type: ignore[arg-type]
                frames=len(snapshots),
                interval=500,
                repeat=True,
            )
            fname = f"route_anim_{instance_name}_{algo}.gif"
            anim.save(anim_dir / fname, writer="pillow", dpi=100)
            plt.close(fig)
            print(f"Animation -> {anim_dir / fname}")


def write_tables(summary: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "summary.csv", index=False)
    with open(out_dir / "summary.tex", "w", encoding="utf-8") as f:
        f.write(summary.to_latex(index=False, float_format="%.2f"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--main-csv", type=str, default="main.csv")
    parser.add_argument("--grid-csv", type=str, default="gridsearch.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--bks", type=Path, default=Path("bks.json"))
    args = parser.parse_args()

    runs_path = args.results_dir / args.main_csv
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} not found.")

    figures_dir = args.results_dir / "figures"
    tables_dir = args.results_dir / "tables"
    history_dir = args.results_dir / "history"

    df = load_runs(runs_path)
    summary = build_summary(df)

    write_tables(summary, tables_dir)
    plot_boxplots(df, figures_dir)
    plot_convergence(df, history_dir, figures_dir)
    plot_delta_vs_n(df, figures_dir)
    plot_best_routes(df, args.data_dir, args.bks, figures_dir)
    plot_route_animation(df, history_dir, args.data_dir, args.bks, figures_dir)
    plot_gridsearch_heatmaps(args.results_dir / args.grid_csv, figures_dir)

    print(f"Analiza zakończona. Tabele: {tables_dir}, wykresy: {figures_dir}")


if __name__ == "__main__":
    main()
