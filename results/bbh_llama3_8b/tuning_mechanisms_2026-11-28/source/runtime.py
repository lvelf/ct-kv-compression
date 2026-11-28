"""Resolve the pinned HF snapshot, validate the environment and preflight context lengths."""
import argparse
import csv
import json
import os
import sys
from functools import lru_cache
from pathlib import Path

from bbh_eval.run_llama_icl import MODEL, MODEL_REV, encode_example, encode_instruction

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
VERSIONS = {
    "transformers": "4.47.1",
    "accelerate": "1.1.1",
    "huggingface-hub": "0.36.0",
    "numpy": "1.26.4",
    "tokenizers": "0.21.4",
    "safetensors": "0.8.0",
}


def packages():
    from importlib.metadata import version

    if sys.version_info[:2] != (3, 10):
        raise SystemExit(f"Python 3.10 required for the pinned study, found {sys.version.split()[0]}")
    for name, expected in VERSIONS.items():
        actual = version(name)
        if actual != expected:
            raise SystemExit(f"{name}: found {actual}, expected {expected}")
    import torch
    if torch.__version__ != "2.4.0+cu121" or torch.version.cuda != "12.1":
        raise SystemExit(f"Expected torch 2.4.0+cu121, found {torch.__version__} (CUDA {torch.version.cuda})")


def gpu():
    import torch

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise SystemExit("An available BF16 CUDA GPU is required. Run nvidia-smi to check the instance.")
    info = torch.cuda.get_device_properties(0)
    print(f"GPU {info.name}; {info.total_memory / 2**30:.1f} GiB; torch {torch.__version__}")
    if "A100" not in info.name or info.total_memory < 78 * 2**30:
        print("Hardware differs from the original A100 80GB; compare wall time and memory accordingly.", file=sys.stderr)


def snapshot_complete(path):
    path = Path(path)
    if path.name != MODEL_REV:
        return False
    required = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
                "model.safetensors.index.json"]
    if not all((path / file).is_file() for file in required):
        return False
    try:
        shards = set(json.loads((path / required[-1]).read_text())["weight_map"].values())
        if not shards or any(not (path / file).is_file() or (path / file).stat().st_size == 0 for file in shards):
            return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def model_path(download):
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError, LocalEntryNotFoundError

    repo_dir = "models--" + MODEL.replace("/", "--")
    paths = []
    if os.environ.get("BBH_SNAPSHOT"):
        paths.append(Path(os.environ["BBH_SNAPSHOT"]))
    # Hugging Face may have been configured through any of these in prior runs.
    if os.environ.get("HF_HUB_CACHE"):
        paths.append(Path(os.environ["HF_HUB_CACHE"]) / repo_dir / "snapshots" / MODEL_REV)
    for home in [Path(os.environ["HF_HOME"]), Path("/workspace/bbh-model-cache"),
                 Path.home() / ".cache" / "huggingface"]:
        paths.append(home / "hub" / repo_dir / "snapshots" / MODEL_REV)
    for path in paths:
        if snapshot_complete(path):
            return path.resolve()
    kwargs = dict(
        repo_id=MODEL, revision=MODEL_REV,
        allow_patterns=["config.json", "generation_config.json", "tokenizer*.json", "special_tokens_map.json",
                        "model*.safetensors", "model.safetensors.index.json"],
    )
    try:
        path = Path(snapshot_download(**kwargs, local_files_only=True))
        if snapshot_complete(path):
            return path.resolve()
    except LocalEntryNotFoundError:
        pass
    if not download:
        raise SystemExit("The exact model snapshot is not cached. Run bash study.sh access.")
    try:
        path = Path(snapshot_download(**kwargs))
    except HfHubHTTPError as exc:
        if not isinstance(exc, GatedRepoError) and getattr(exc.response, "status_code", None) not in (401, 403):
            raise
        from getpass import getpass
        if not sys.stdin.isatty():
            raise SystemExit("Run bash study.sh access in an interactive terminal or provide HF_TOKEN in your environment.")
        token = getpass("Approved Hugging Face read token (hidden): ").strip()
        path = Path(snapshot_download(**kwargs, token=token))
    if not snapshot_complete(path):
        raise SystemExit("The downloaded pinned snapshot is incomplete. Run bash study.sh access again.")
    return path.resolve()


def preflight(model):
    from transformers import AutoConfig, AutoTokenizer
    from bbh_eval.prepare_benchmark_splits import benchmark_rows
    from design import build_jobs

    config = AutoConfig.from_pretrained(model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True, local_files_only=True)
    limit = int(config.max_position_embeddings)
    specs = json.loads((ROOT / "data/manifest.json").read_text())["tasks"]

    @lru_cache(None)
    def data(task):
        return benchmark_rows(task, ROOT / "data/bbh")

    @lru_cache(None)
    def demo_length(task, i):
        r = data(task)[i]
        sequence, _, end = encode_example(tokenizer, r["input"], r["target"])
        return len(sequence), end - 1

    @lru_cache(None)
    def query_length(task, i):
        return len(encode_example(tokenizer, data(task)[i]["input"])[0])

    rows = []
    for job in build_jobs():
        task = job["task"]
        spec = tuple(specs[task][x] for x in ("task_prompt", "answer_format", "generation_length"))
        instruction = len(encode_instruction(tokenizer, task, spec))
        demos, test, valid = job["demo_ids"], job["test_ids"], job["val_ids"]
        prefix = instruction + sum(demo_length(task, i)[0] for i in demos)
        test_position = prefix + max(query_length(task, i) for i in test) + spec[-1]
        train_position = prefix + max(demo_length(task, i)[1] for i in demos) if job["updates"] else 0
        # Full prefix positions remain relevant after physical cache compaction (RoPE).
        val_position = 0
        if valid:
            val_position = prefix + max(max(query_length(task, i) + spec[-1], demo_length(task, i)[1]) for i in valid)
        fits = max(test_position, train_position, val_position) <= limit
        rows.append(dict(id=job["id"], task=task, prefix_tokens=prefix, max_test_positions=test_position,
                         max_training_positions=train_position, max_validation_positions=val_position,
                         context_limit=limit, fits_context=fits))
    if not rows:
        raise SystemExit("Plan returned no configurations")
    RESULTS.mkdir(parents=True, exist_ok=True)
    out = RESULTS / "preflight.csv"
    part = out.with_suffix(".csv.part")
    with part.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    part.replace(out)
    failed = [r["id"] for r in rows if not r["fits_context"]]
    if failed:
        raise SystemExit(f"{len(failed)} exceed the context limit, first: {failed[0]}. See {out}")
    print(f"All {len(rows)} configurations fit {limit} context positions; saved {out}")


def main():
    args = argparse.ArgumentParser(description=__doc__)
    args.add_argument("action", choices=("packages", "gpu", "access", "path", "preflight"))
    a = args.parse_args()
    if a.action == "packages":
        packages()
    elif a.action == "gpu":
        gpu()
    elif a.action == "preflight":
        packages()
        gpu()
        preflight(model_path(False))
    else:
        print(model_path(a.action == "access"))


if __name__ == "__main__":
    main()
