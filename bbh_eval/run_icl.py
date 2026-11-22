"""Run a greedy text-prompt ICL baseline on BBH development questions."""

import argparse
import hashlib
import json
from pathlib import Path

from bbh_eval.prepare_splits import (
    DEFAULT_MANIFEST, DEMO_SEEDS, REVISION, TASK_HASHES, load_examples,
    make_task_split, write_atomically,
)
from bbh_eval.score import score_answer


DEFAULT_MODEL = "openai-community/gpt2-large"
DEFAULT_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
TASKS = ("boolean_expressions", "date_understanding")
PROMPT_VERSION = "direct-qa-v1"


def make_prefix(rows, demo_ids):
    demonstrations = "".join(
        f"Q: {rows[i]['input']}\nA: {rows[i]['target']}\n\n" for i in demo_ids
    )
    return "Answer each question directly.\n\n" + demonstrations


def write_metadata(directory, metadata):
    write_atomically(directory / "metadata.json", (json.dumps(metadata, indent=2) + "\n").encode())


def select_run_ids(split, shots, limit):
    """Take the first shots in manifest order, leaving the development split fixed."""
    return split["demo_ids"][:shots], split["dev_ids"][:limit]


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, choices=DEMO_SEEDS, default=0, help="Demonstration/development seed")
    parser.add_argument("--shots", type=int, choices=(8, 10), default=10,
                        help="Use the first 8 or all 10 listed demonstrations (default: 10); keep the same dev IDs")
    parser.add_argument("--out", type=Path, required=True, help="A new directory for this one run")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--source-dir", type=Path, help="Local folder containing the pinned BBH files")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Hugging Face model ID or local checkpoint (default: {DEFAULT_MODEL})")
    parser.add_argument("--revision", help="Hugging Face model revision; the default GPT-2 large model is pinned")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), help="Defaults to CUDA if available, then CPU")
    parser.add_argument("--limit", type=int, choices=range(1, 41), metavar="1..40", help="Evaluate only the first N dev IDs")
    parser.add_argument("--max-new-tokens", type=int, default=12)
    return parser


def main():
    parser = make_parser()
    args = parser.parse_args()
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be positive")
    if args.out.exists():
        parser.error(f"Output directory already exists: {args.out}. Choose a new run name.")
    if not args.manifest.is_file():
        parser.error("Missing split manifest. Run python3 -m bbh_eval.prepare_splits first.")
    manifest = json.loads(args.manifest.read_text())
    rows = load_examples(args.task, args.source_dir)
    task_split = manifest["tasks"][args.task]
    expected = {"sha256": TASK_HASHES[args.task], **make_task_split(len(rows))}
    if manifest["dataset_revision"] != REVISION or task_split != expected:
        parser.error("The dataset revision or split IDs differ from the pinned protocol.")
    split = task_split["seeds"][str(args.seed)]
    demo_ids, dev_ids = select_run_ids(split, args.shots, args.limit)
    prefix = make_prefix(rows, demo_ids)

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    revision = args.revision or (DEFAULT_REVISION if args.model == DEFAULT_MODEL else None)
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(args.model, revision=revision, torch_dtype=dtype)
    model.to(device).eval()
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    context_limit = getattr(model.config, "max_position_embeddings", None)

    class StopAtNewline(StoppingCriteria):
        """Single-question greedy decoding; only inspect tokens after the prompt."""
        def __init__(self, start):
            self.start = start

        def __call__(self, input_ids, scores, **kwargs):
            generated = tokenizer.decode(input_ids[0, self.start:], skip_special_tokens=True)
            return "\n" in generated

    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "status": "running", "method": "text_icl", "task": args.task, "evaluation_split": "development",
        "demo_seed": args.seed, "n_shots": len(demo_ids), "demo_ids": demo_ids, "dev_ids": dev_ids,
        "n_full_dev": len(split["dev_ids"]), "dataset_revision": REVISION,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "model": args.model, "requested_revision": revision,
        "resolved_revision": getattr(model.config, "_commit_hash", None),
        "device": device, "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "dtype": str(dtype), "torch": str(torch.__version__), "transformers": transformers.__version__,
        "prompt_version": PROMPT_VERSION, "chat_template": False,
        "prefix_tokens": len(tokenizer.encode(prefix, add_special_tokens=False)),
        "prefix_sha256": hashlib.sha256(prefix.encode()).hexdigest(),
        "do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens,
        "answer_policy": "One line ending at newline, EOS or token cap. Strip outer whitespace for exact match.",
    }
    write_metadata(args.out, metadata)
    correct = invalid = 0
    with (args.out / "predictions.jsonl").open("x", encoding="utf-8") as log, torch.inference_mode():
        for number, row_id in enumerate(dev_ids, 1):
            question = rows[row_id]["input"]
            prompt = prefix + f"Q: {question}\nA:"
            encoded = tokenizer(
                prompt, return_tensors="pt", add_special_tokens=False, return_token_type_ids=False,
            ).to(device)
            prompt_len = encoded["input_ids"].shape[1]
            if context_limit and prompt_len + args.max_new_tokens > context_limit:
                raise ValueError(f"Row {row_id} plus answer budget exceeds the model context window. No truncation applied.")
            output = model.generate(
                **encoded, max_new_tokens=args.max_new_tokens, do_sample=False, num_beams=1,
                use_cache=True, pad_token_id=pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopAtNewline(prompt_len)]),
            )
            new_ids = output[0, prompt_len:].tolist()
            raw_text = tokenizer.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            answer = raw_text.partition("\n")[0]
            # The current question's gold label enters only after generation, for scoring.
            gold = rows[row_id]["target"]
            score = score_answer(args.task, answer, gold, question)
            record = {
                "row_id": row_id, "input": question, "gold": gold, "raw_generated_text": raw_text,
                "new_token_ids": new_ids, "prompt_tokens": prompt_len, **score,
            }
            print(json.dumps(record, ensure_ascii=False), file=log, flush=True)
            correct += int(score["exact_match"])
            invalid += int(not score["valid_format"])
            print(f"dev {number}/{len(dev_ids)}, correct {correct}, invalid {invalid}", flush=True)
    metadata.update(status="complete", n_scored=len(dev_ids), n_correct=correct, n_invalid=invalid,
                    exact_match_accuracy=correct / len(dev_ids))
    write_metadata(args.out, metadata)
    print(f"Results: {args.out}")


if __name__ == "__main__":
    main()
