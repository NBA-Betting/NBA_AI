#!/usr/bin/env python3
"""
Promote a retrained Phase5 model into production.

Training writes to checkpoints/ and data/ (L1 cache, L1 vectors); live
predictions read models/phase5/ and the L1 cache/vectors. This copies one to
the other, after backing up what it replaces, so production only changes when
a retrained model has passed the backtest (scripts/backtest.py).

Copies (source root -> target root, default both this checkout):
  checkpoints/phase5/phase2_best.pt  -> models/phase5/l1.pt
  checkpoints/phase5/l2_best.pt      -> models/phase5/l2.pt
  checkpoints/phase5/phase_c_best.pt -> models/phase5/l3l4.pt
  data/phase5_cache/{metadata.json, profiles.npz, archetypes.npz}
  data/l2_cache/l1_vectors/          (replaced as a whole)

Usage:
    python scripts/promote_phase5.py --source ../NBA_AI-retrain [--dry-run]
"""

import argparse
import shutil
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FILES = {
    "checkpoints/phase5/phase2_best.pt": "models/phase5/l1.pt",
    "checkpoints/phase5/l2_best.pt": "models/phase5/l2.pt",
    "checkpoints/phase5/phase_c_best.pt": "models/phase5/l3l4.pt",
    "data/phase5_cache/metadata.json": "data/phase5_cache/metadata.json",
    "data/phase5_cache/profiles.npz": "data/phase5_cache/profiles.npz",
    "data/phase5_cache/archetypes.npz": "data/phase5_cache/archetypes.npz",
}
VECTORS = "data/l2_cache/l1_vectors"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--source", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--target", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    src, dst = args.source.resolve(), args.target.resolve()

    missing = [s for s in [*FILES, VECTORS] if not (src / s).exists()]
    if missing:
        raise SystemExit(f"Missing in {src}: {', '.join(missing)}")

    backup = (
        dst / "data" / "backups" / f"phase5_promote_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    # Within one checkout the caches are already in place (source is target)
    copies = [
        (s, t)
        for s, t in [*FILES.items(), (VECTORS, VECTORS)]
        if (src / s).resolve() != (dst / t).resolve()
    ]
    for s, t in copies:
        print(f"{src / s} -> {dst / t}")
    print(f"Backup of replaced files: {backup}")
    if args.dry_run:
        return

    for _, t in copies:
        if (dst / t).exists():
            (backup / t).parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dst / t), str(backup / t))
    for s, t in copies:
        (dst / t).parent.mkdir(parents=True, exist_ok=True)
        if t == VECTORS:
            shutil.copytree(src / s, dst / t)
        else:
            shutil.copy2(src / s, dst / t)
    print("Promoted. Restore by moving the backup files back.")


if __name__ == "__main__":
    main()
