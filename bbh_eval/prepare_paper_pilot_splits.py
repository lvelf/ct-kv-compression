"""Extend the team's four-task BBH development pilot to five local demonstration seeds."""

import argparse
import json
import random
from pathlib import Path

from bbh_eval.prepare_splits import (
    DEFAULT_MANIFEST, DEMO_SEEDS, REVISION, TASK_HASHES, TEST_SEED,
    make_task_split, write_atomically,
)


def build_manifest(team):
    if team["dataset_revision"] != REVISION or team["test_seed"] != TEST_SEED:
        raise ValueError("Unexpected source data or team split seed.")
    if set(team["tasks"]) != set(TASK_HASHES):
        raise ValueError("Expected exactly the four proposal tasks.")
    for task, digest in TASK_HASHES.items():
        entry = team["tasks"][task]
        if entry != {"sha256": digest, **make_task_split(250)}:
            raise ValueError(f"{task}: input does not match the team's committed split.")
        pool = sorted(set(range(250)) - set(entry["test_ids"]))
        for seed in range(5):
            if seed in DEMO_SEEDS:
                continue
            demo = sorted(random.Random(seed).sample(pool, 10))
            entry["seeds"][str(seed)] = {
                "demo_ids": demo,
                "dev_ids": sorted(set(pool) - set(demo)),
            }
    return team


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_MANIFEST.with_name("paper_pilot_splits.json"))
    args = parser.parse_args()
    if args.output.resolve() == args.base.resolve():
        parser.error("Choose a separate output path to preserve the three-seed team manifest.")
    result = build_manifest(json.loads(args.base.read_text()))
    data = (json.dumps(result, indent=2) + "\n").encode()
    if args.output.exists() and args.output.read_bytes() != data:
        parser.error(f"An unrelated manifest already exists at {args.output}. Choose a new path.")
    if not args.output.exists():
        write_atomically(args.output, data)
    print("Four-task pilot: local demo seeds 0-4, 10 demos and 40 dev rows per seed.")
    print("Seeds 0-2 and all 200 held-out test IDs per task match the team manifest.")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
