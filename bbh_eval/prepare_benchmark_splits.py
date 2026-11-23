"""Build a reconstructed BBH-wide protocol from pinned BBH data and Fewshot-TTT task specs.

The five local seeds and the split policy come from this reconstruction. They are
not verified as the ordered demonstration IDs of the Context Tuning paper.
"""
import argparse
import ast
import hashlib
import json
import random
from pathlib import Path
from urllib.request import urlopen

from bbh_eval.prepare_splits import (
    DEFAULT_DATA_DIR, REVISION, SOURCE_URL, TASK_HASHES, make_task_split, write_atomically,
)


BENCHMARK_PROTOCOL = "antecedent_bbh27_v1"
SPEC_REVISION = "ad40084c63f5cf503e59d2acdab426843dffd254"
SPEC_URL = f"https://raw.githubusercontent.com/adamzweiger/Fewshot-TTT/{SPEC_REVISION}/src/tasks.py"
SPEC_SHA = "ab8f2ecefb71a4d2461220c44d0f585a3e9bb6898cce44390d6036721a4de75f"
BENCHMARK_SEEDS = (0, 1, 2, 3, 4)
SPLIT_POLICIES = ("antecedent", "project_demo_pool", "protect_project_test")

# SHA-256 byte digests for bbh/*.json at the BBH revision pinned in prepare_splits.
BENCHMARK_HASHES = {
    "boolean_expressions": "ea6c754ec005e2d3f2d085d349a740f593b5764f32ea4638ffed4cfc0061b12a",
    "causal_judgement": "6e8bca44dd402008ed41c07a3d0a1d5e7b03db7001a7e002fcdf4c60985ea081",
    "date_understanding": "0148d4ac5fca05b2f82373e5fef7208e15363b9c6079493034d77b3dab496bf5",
    "disambiguation_qa": "cf06df9ebf911b883671cb1b35d3fbd1a1c306b19b9e9a012407e596c85fe8c6",
    "dyck_languages": "3f9badedf15c2be1ce53d94fdf016299bb30ba231ef087ab9f51a7ed6d0bdc25",
    "formal_fallacies": "4c1adb664821157faade1d4153685403afcc02648f3c180a608a18f17f589629",
    "geometric_shapes": "07be68a124420caa2ebc0b2c858ad79fc55c5c8806170e496e1e17ebfb11f31c",
    "hyperbaton": "b6d7f55c75f5d02f2987a4f3123802dfa093872089a56a334144aae8a7264fdc",
    "logical_deduction_five_objects": "d2df31394b903b7f0e085d42a69572a092987856eac076799017d70ef266fa8f",
    "logical_deduction_seven_objects": "2896c7e3482eea318dd37bcc370d24ec3cc91e8374c1784287b5dbd38a529e33",
    "logical_deduction_three_objects": "e590fae72e3a6041fbd23bd8ff6b74baf44ca563f6ba40a041c55e7b867c1681",
    "movie_recommendation": "a8b53a87ee21a865b7248e694f388a9b8f7cf6b839b11d44796a2dd3fc088983",
    "multistep_arithmetic_two": "c9d5a6433c9b78256422a15c78463949c8bec8d85607aa28b52d7234ad4a8595",
    "navigate": "b7944dfd3c81da5c4815bd75b00d6fa3c64140823498e78ed496f4c3d36f7f0d",
    "object_counting": "8acef14dbbdd40d2326ef7755750b6b5fe919fef06276af1bab3f1c297f022ff",
    "penguins_in_a_table": "bb87ee449698eee51215e025a453b1bcb2f5e6589c2063b2c79e68b8f0655228",
    "reasoning_about_colored_objects": "0a8850cf95166020fce7a78244026ceedd7ea839302471512b4f8744e85a6a49",
    "ruin_names": "b6f723aef2bafdd4c7bfc3c7e0329c7e8a8279391a609969c8ed76c9319f5f7f",
    "salient_translation_error_detection": "becf3eb8dd53c821a555ecd5c1334ecd93eb25f980e63be963c518e7138de5da",
    "snarks": "54a0661fe2de06167645a4965e6bcf963c4ba307e76262a7c750998b0e3d0d0e",
    "sports_understanding": "63ca262858f6176864515cd8cf53cdaa40839ebecc155b726d77930855d167bf",
    "temporal_sequences": "92a1189faaa84aacc4c405aaa13f13f48c2992eb884772eedb622505252baef5",
    "tracking_shuffled_objects_five_objects": "2063a46a4fb484236ed4310523606543a3e1dd4a4dc34eda65d407910ece25a5",
    "tracking_shuffled_objects_seven_objects": "b59fd05b850c37cf57cb514ae1e6975c00b1638ade2291722056b2af3c54084d",
    "tracking_shuffled_objects_three_objects": "546cc7bf10c6a01a6a4c17ae28a4966485f21f29fd5d111f37858ec78e354f6b",
    "web_of_lies": "3e60323f8514a107fdb4edf35f8832a9171c5a09c0eaa47823559de905e8ab8b",
    "word_sorting": "2a6132d2c99f00d0d2eb1113ac6b4a918bd969d863c48749210d969713db8d43",
}


def read_verified(path, expected, url=None):
    if path.exists():
        raw = path.read_bytes()
    elif url is not None:
        with urlopen(url, timeout=60) as resp:
            raw = resp.read()
    else:
        raise FileNotFoundError(path)
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError(f"Unexpected SHA-256 for {path}; expected {expected}")
    if not path.exists():
        write_atomically(path, raw)
    return raw


def benchmark_rows(task, source_dir=None):
    digest = BENCHMARK_HASHES[task]
    folder = DEFAULT_DATA_DIR if source_dir is None else Path(source_dir)
    url = f"{SOURCE_URL}/{task}.json" if source_dir is None else None
    rows = json.loads(read_verified(folder / f"{task}.json", digest, url))["examples"]
    return rows


def load_task_specs(spec_source=None):
    if spec_source is not None:
        raw = read_verified(Path(spec_source), SPEC_SHA)
    else:
        path = DEFAULT_DATA_DIR.parent / "fewshot-ttt" / SPEC_REVISION / "tasks.py"
        raw = read_verified(path, SPEC_SHA, SPEC_URL)
    # Read the literal dictionary only. The referenced module is never imported or run.
    for stmt in ast.parse(raw).body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name) and stmt.targets[0].id == "TASKS"):
            specs = ast.literal_eval(stmt.value)
            if set(specs) != set(BENCHMARK_HASHES):
                raise ValueError("Antecedent task inventory does not match the pinned BBH inputs")
            return specs
    raise ValueError("No literal TASKS dictionary found")


def benchmark_split(n_rows, seed, task=None, policy="antecedent"):
    if policy not in SPLIT_POLICIES:
        raise ValueError(f"Unknown benchmark split policy: {policy}")
    order = list(range(n_rows))
    random.Random(seed).shuffle(order)
    if task not in TASK_HASHES or policy == "antecedent":
        return order[:10], order[10:]
    committed = make_task_split(n_rows)
    pool = sorted(set(range(n_rows)) - set(committed["test_ids"]))
    demos = sorted(random.Random(seed).sample(pool, 10))
    evaluation_pool = pool if policy == "protect_project_test" else order
    return demos, [i for i in evaluation_pool if i not in demos]


def build_manifest(spec_source=None, source_dir=None, policy="antecedent"):
    specs = load_task_specs(spec_source)
    tasks = {}
    for task, spec in specs.items():
        rows = benchmark_rows(task, source_dir)
        seeds = {}
        for seed in BENCHMARK_SEEDS:
            demo, test = benchmark_split(len(rows), seed, task, policy)
            seen = {rows[i]["input"] for i in demo}
            overlaps = [i for i in test if rows[i]["input"] in seen]
            seeds[str(seed)] = {
                "demo_ids": demo, "test_ids": test,
                "test_ids_with_duplicate_demo_text": overlaps,
            }
        tasks[task] = {
            **spec, "sha256": BENCHMARK_HASHES[task], "n_rows": len(rows), "seeds": seeds,
        }
    return {"protocol": BENCHMARK_PROTOCOL, "split_policy": policy, "dataset_revision": REVISION,
            "antecedent_revision": SPEC_REVISION, "task_spec_sha256": SPEC_SHA,
            "seeds": list(BENCHMARK_SEEDS), "tasks": tasks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Offline folder of all 27 pinned BBH JSON files.")
    parser.add_argument("--spec-source", type=Path, help="Offline copy of the pinned Fewshot-TTT src/tasks.py.")
    parser.add_argument("--output", type=Path, required=True)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--project-demo-pool", dest="policy", action="store_const",
                       const="project_demo_pool",
                       help="Use the project demo pool on the four proposal tasks; evaluate remaining rows.")
    group.add_argument("--protect-project-test", dest="policy", action="store_const",
                       const="protect_project_test",
                       help="Use only the 50 development-pool rows for the four proposal tasks.")
    parser.set_defaults(policy="antecedent")
    args = parser.parse_args()
    manifest = build_manifest(args.spec_source, args.source_dir, args.policy)
    data = (json.dumps(manifest, indent=2) + "\n").encode()
    if args.output.exists():
        if args.output.read_bytes() != data:
            parser.error(f"Different manifest already exists: {args.output}")
    else:
        write_atomically(args.output, data)
    tasks = manifest["tasks"]
    per_seed = sum(len(x["seeds"]["0"]["test_ids"]) for x in tasks.values())
    print(f"Split policy: {args.policy}")
    print(f"{len(tasks)} task files; {per_seed} evaluation rows per seed; seeds {BENCHMARK_SEEDS}.")
    for task, entry in tasks.items():
        for seed, part in entry["seeds"].items():
            if part["test_ids_with_duplicate_demo_text"]:
                print(f"Source has identical demo/test text: {task}, seed {seed}, "
                      f"test IDs {part['test_ids_with_duplicate_demo_text']}")
    print(f"Manifest: {args.output}")


if __name__ == "__main__":
    main()
