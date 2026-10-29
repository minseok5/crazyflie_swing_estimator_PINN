#!/usr/bin/env python3
"""Generate independent simulation episodes in parallel.

Run from repository root:
    python -m dataset_generation.generate_dataset_parallel --episodes 4 --workers 2 --duration 10
"""
from __future__ import annotations

import argparse
import csv
import math
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

from . import dataset_config as cfg
from .sim_payload_dataset import ensure_model, run_episode


def build_episode_configs(episodes: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    configs = []
    angles = cfg.INITIAL_SWING_ANGLES_DEG
    for i in range(1, episodes + 1):
        swing_x = math.radians(rng.choice(angles))
        swing_y = math.radians(rng.choice(angles))
        trajectory = rng.choice(cfg.TRAJECTORIES)
        speed_scale = rng.choice(cfg.SPEED_SCALES)
        configs.append({
            "episode_id": i,
            "trajectory": trajectory,
            "initial_swing_x_rad": swing_x,
            "initial_swing_y_rad": swing_y,
            "speed_scale": speed_scale,
            "seed": seed + i,
        })
    return configs


def _run_worker(kwargs: dict) -> dict:
    return run_episode(**kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=cfg.DEFAULT_EPISODES)
    parser.add_argument("--workers", type=int, default=cfg.DEFAULT_WORKERS)
    parser.add_argument("--duration", type=float, default=cfg.DEFAULT_DURATION)
    parser.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    parser.add_argument("--output-dir", type=str, default=str(cfg.OUTPUT_DIR))
    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.duration <= 0:
        parser.error("--duration must be positive")

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = cfg.PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate the model before starting workers to avoid concurrent generation.
    ensure_model()
    configs = build_episode_configs(args.episodes, args.seed)
    jobs = [
        {
            **item,
            "output_dir": str(output_dir),
            "duration": args.duration,
        }
        for item in configs
    ]

    results = []
    errors = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_worker, job): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(
                    f"[{result['episode_id']:04d}/{args.episodes:04d}] "
                    f"{result['status']} | {result['csv_path']}"
                )
            except Exception as exc:
                errors.append({
                    "episode_id": job["episode_id"],
                    "status": "error",
                    "error": repr(exc),
                    **{k: job[k] for k in (
                        "trajectory", "initial_swing_x_rad",
                        "initial_swing_y_rad", "speed_scale", "seed",
                    )},
                })
                print(f"[{job['episode_id']:04d}] ERROR: {exc}")

    manifest_path = output_dir / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    fields = [
        "episode_id", "csv_path", "status", "trajectory", "duration_s",
        "initial_swing_x_rad", "initial_swing_y_rad", "speed_scale", "seed", "error",
    ]
    all_rows = []
    for result in results:
        all_rows.append({**result, "error": ""})
    all_rows.extend(errors)
    all_rows.sort(key=lambda row: row["episode_id"])
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\nManifest saved: {manifest_path}")
    print(f"Completed: {len(results) - len([r for r in results if r['status'] != 'completed'])}")
    print(f"Other statuses: {sum(r['status'] != 'completed' for r in results)}")
    print(f"Errors: {len(errors)}")


if __name__ == "__main__":
    main()
