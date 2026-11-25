"""Prepare the team's four BBH tasks. Run from the repo root with Python 3.9+."""

import argparse
import hashlib
import json
import random
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlopen


REVISION = "9ee07bd481feebf959a6b59d61ea57bdcf30964d"
SOURCE_URL = f"https://raw.githubusercontent.com/suzgunmirac/BIG-Bench-Hard/{REVISION}/bbh"
TASK_HASHES = {
    "boolean_expressions": "ea6c754ec005e2d3f2d085d349a740f593b5764f32ea4638ffed4cfc0061b12a",
    "date_understanding": "0148d4ac5fca05b2f82373e5fef7208e15363b9c6079493034d77b3dab496bf5",
    "word_sorting": "2a6132d2c99f00d0d2eb1113ac6b4a918bd969d863c48749210d969713db8d43",
    "logical_deduction_three_objects": "e590fae72e3a6041fbd23bd8ff6b74baf44ca563f6ba40a041c55e7b867c1681",
}
TEST_SEED = 4705
DEMO_SEEDS = (0, 1, 2)
DEFAULT_MANIFEST = Path(__file__).resolve().with_name("splits.json")
DEFAULT_DATA_DIR = DEFAULT_MANIFEST.parent.parent / ".cache" / "bbh" / REVISION


def write_atomically(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        tmp.write(data)
    Path(tmp.name).replace(path)


def load_examples(task, source_dir=None):
    """Load the pinned source rows and check their hash and example count."""
    if task not in TASK_HASHES:
        raise ValueError(f"Unknown BBH task: {task}")
    folder = DEFAULT_DATA_DIR if source_dir is None else Path(source_dir)
    path = folder / f"{task}.json"
    download = source_dir is None and not path.is_file()
    if download:
        with urlopen(f"{SOURCE_URL}/{task}.json", timeout=60) as response:
            raw = response.read()
    else:
        raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != TASK_HASHES[task]:
        raise ValueError(f"{path.name} does not match the official file at {REVISION}.")
    rows = json.loads(raw)["examples"]
    if len(rows) != 250 or len({row["input"] for row in rows}) != 250:
        raise ValueError(f"{task}: expected 250 unique questions.")
    if download:
        write_atomically(path, raw)
    return rows


def make_task_split(row_count):
    permutation = random.Random(TEST_SEED).sample(range(row_count), row_count)
    test = permutation[:200]
    pool = sorted(permutation[200:])
    seed_splits = {}
    for seed in DEMO_SEEDS:
        demo = sorted(random.Random(seed).sample(pool, 10))
        dev = sorted(set(pool) - set(demo))
        if (len(test), len(demo), len(dev)) != (200, 10, 40):
            raise ValueError("Incorrect split size.")
        if set(test + demo + dev) != set(range(row_count)):
            raise ValueError("The partition overlaps or omits questions.")
        seed_splits[str(seed)] = {"demo_ids": demo, "dev_ids": dev}
    return {"test_ids": test, "seeds": seed_splits}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Read BBH task JSONs from this folder instead of downloading.")
    parser.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args()
    manifest = {"dataset_revision": REVISION, "test_seed": TEST_SEED, "tasks": {}}
    for task, sha256 in TASK_HASHES.items():
        rows = load_examples(task, args.source_dir)
        manifest["tasks"][task] = {"sha256": sha256, **make_task_split(len(rows))}
        print(f"{task}: 250 unique questions; 200 test, 10 demos + 40 dev for each of 3 seeds")
    write_atomically(args.output, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(f"Saved split IDs to {args.output}")


if __name__ == "__main__":
    main()
