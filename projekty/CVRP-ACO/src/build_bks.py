from __future__ import annotations

import argparse
from pathlib import Path

from .utils import build_bks_from_sol_files


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("bks.json"))
    args = parser.parse_args()
    bks = build_bks_from_sol_files(args.data_dir, args.output)
    print(f"BKS for {len(bks)} instances -> {args.output}")


if __name__ == "__main__":
    main()
