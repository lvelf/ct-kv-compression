"""Evaluate one registered BBH mechanisms job using the pinned 144-run scorer."""
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
from time import perf_counter

from bbh_eval.prepare_benchmark_splits import BENCHMARK_HASHES, SPEC_SHA, benchmark_rows
from bbh_eval.prepare_splits import REVISION, write_atomically
from bbh_eval.run_llama_icl import (
    MODEL, MODEL_REV, PROMPT_SOURCE, TASKS, antecedent_match, encode_example,
    encode_instruction, greedy_answer, score,
)

HERE = Path(__file__).resolve().parent
MANIFEST = HERE / "data" / "manifest.json"


def write_json(path, obj):
    write_atomically(path, (json.dumps(obj, indent=2, allow_nan=False) + "\n").encode())


def load_inputs(job):
    task = job["task"]
    shots = job["shots"]
    method = job["method"]
    mode = job["param_mode"]
    budget = job["budget"]
    updates = job["updates"]
    if task not in {"date_understanding", "boolean_expressions"} or shots not in (10, 20):
        raise ValueError("Unexpected BBH task or shot count")
    if method not in {"icl", "full", "answer_attention", "random"} or not 0 < budget <= 1:
        raise ValueError("Unexpected method or KV budget")
    if (method in {"icl", "full"}) != (budget == 1):
        raise ValueError("Full retention requires ICL or full CT-KV, compression requires a selection method")
    if updates < 0 or (method == "icl" and updates) or mode not in {"both", "key", "value"}:
        raise ValueError("Invalid updates or parameter mode")
    raw = MANIFEST.read_bytes()
    manifest = json.loads(raw)
    cfg = manifest["tasks"][task]
    if manifest["dataset_revision"] != REVISION or manifest["task_spec_sha256"] != SPEC_SHA or cfg["sha256"] != BENCHMARK_HASHES[task]:
        raise ValueError("The data revision or task spec differs from the original study")
    spec = (cfg["task_prompt"], cfg["answer_format"], cfg["generation_length"])
    if spec != TASKS[task]:
        raise ValueError("BBH prompt or decoding limit changed")
    rows = benchmark_rows(task, HERE / "data" / "bbh")
    demos, test, val = (job[key] for key in ("demo_ids", "test_ids", "val_ids"))
    if len(demos) != shots or any(len(ids) != len(set(ids)) for ids in (demos, test, val)):
        raise ValueError("Duplicate row IDs or wrong number of demonstrations")
    support = set(cfg["support_pool_ids"])
    if not set(demos + val) <= support or set(demos) & set(val) or support & set(test):
        raise ValueError("Split leakage across demonstrations, validation, or test")
    if not test or not val:
        raise ValueError("Empty evaluation split")
    expected_test = cfg["test_ids"][:len(test)] if job.get("smoke") else cfg["test_ids"]
    if test != expected_test or (not job.get("smoke") and len(test) != 200):
        raise ValueError("Fixed test question IDs or ordering changed")
    sets = [{rows[i]["input"] for i in ids} for ids in (demos, val, test)]
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError("Question text overlaps across splits")
    ratio = job["ratio"]
    if ratio is None:
        if demos != cfg["seeds"][str(job["demo_seed"])]["demo_ids"][:shots]:
            raise ValueError("Original demonstration IDs or ordering changed")
    elif task != "boolean_expressions" or sum(rows[i]["target"] == "True" for i in demos) != round(shots * ratio):
        raise ValueError("Stratified demonstration labels do not match the requested ratio")
    return rows, spec, hashlib.sha256(raw).hexdigest()


def prepare(job, tok, limit):
    rows, spec, manifest_hash = load_inputs(job)
    task = job["task"]
    ids = job["demo_ids"]
    instruction = encode_instruction(tok, task, spec)
    demos = [encode_example(tok, rows[i]["input"], rows[i]["target"]) for i in ids]
    prefix_ids = instruction + [token for tokens, _, _ in demos for token in tokens]
    prefix_text = f"{spec[0]} {spec[1]}\n\n" + "".join(
        f"Q: {rows[i]['input']}\nA: {rows[i]['target']}\n\n" for i in ids
    )
    val_labels = [encode_example(tok, rows[i]["input"], rows[i]["target"]) for i in job["val_ids"]]
    support_lengths = [end - 1 for _, _, end in demos + val_labels]
    if len(prefix_ids) + max(support_lengths) > limit:
        raise ValueError("Training or validation loss requires more positions than the model supports")
    def questions(which):
        result = []
        for i in job[which]:
            question = rows[i]["input"]
            query = encode_example(tok, question)[0]
            direct = [tok.bos_token_id] + tok.encode(prefix_text + f"Q: {question}\nA:", add_special_tokens=False)
            if prefix_ids + query != direct:
                raise ValueError(f"Tokenization changed across the prompt boundary for row {i}")
            if len(direct) + spec[2] > limit:
                raise ValueError(f"Evaluation exceeds the context window on row {i}")
            result.append((i, question, query, rows[i]["target"]))
        return result
    meta = dict(model=MODEL, revision=MODEL_REV, dataset_revision=REVISION,
                dataset_sha256=BENCHMARK_HASHES[task], manifest_sha256=manifest_hash,
                prompt_source=PROMPT_SOURCE, prompt_protocol="bbh-direct-qa-v1",
                prompt_prefix_sha256=hashlib.sha256(prefix_text.encode()).hexdigest(),
                max_new_tokens=spec[2], context_limit=limit, n_instruction_tokens=len(instruction),
                n_demo_tokens=len(prefix_ids) - len(instruction), n_shots=len(demos))
    if task == "boolean_expressions":
        meta["n_demo_true"] = sum(rows[i]["target"] == "True" for i in ids)
    return spec, instruction, demos, prefix_ids, questions("test_ids"), questions("val_ids"), val_labels, meta


def evaluate(task, answer_format, queries, predict):
    records = []
    for i, question, ids, gold in queries:
        raw, tokens = predict(ids)
        prediction, valid, correct = score(task, question, raw, gold)
        records.append(dict(row_id=i, gold=gold, prediction=prediction, raw_generated_text=raw,
                            generated_token_ids=tokens, exact_match=correct, valid_format=valid,
                            antecedent_exact_match=antecedent_match(raw, gold, answer_format)))
    return records


def scores(task, rows):
    n = len(rows)
    correct = sum(row["exact_match"] for row in rows)
    boolean = task == "boolean_expressions"
    result = dict(n_scored=n, n_correct=correct, accuracy=correct / n,
                  n_invalid=sum(r["valid_format"] is False for r in rows),
                  n_pred_true=sum(r["prediction"] == "True" for r in rows) if boolean else None)
    for label in ("True", "False"):
        selected = [r for r in rows if r["gold"] == label]
        key = label.lower()
        result[f"n_gold_{key}"] = len(selected) if boolean else None
        result[f"accuracy_{key}"] = sum(r["exact_match"] for r in selected) / len(selected) if selected else None
    return result


def run_job(job, out, model, tok):
    """Model and tokenizer are injected to support offline CPU numerical checks."""
    import torch
    import transformers
    from bbh_eval.ctkv import decode, fit, validation_nll
    device = next(model.parameters()).device
    cuda = device.type == "cuda"

    def sync():
        if cuda:
            torch.cuda.synchronize(device)

    spec, inst, demos, direct_ids, test, val, val_labels, meta = prepare(job, tok, model.config.max_position_embeddings)
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any((out / file).exists() for file in ("predictions.jsonl", "metrics.json", "validation.json")):
        raise FileExistsError(f"Refusing to overwrite existing output files in {out}")
    model.eval().requires_grad_(False)
    meta.update(job)
    meta.update(dtype=str(next(model.parameters()).dtype), torch=str(torch.__version__),
                transformers=transformers.__version__, python=platform.python_version(),
                gpu=torch.cuda.get_device_name(device) if cuda else None, cuda=torch.version.cuda,
                device=str(device), attention_backend=model.config._attn_implementation,
                microbatch=2, model_loaded_allocated_bytes=torch.cuda.memory_allocated(device) if cuda else None)
    method = job["method"]
    cap = spec[2]
    mode = job["param_mode"]
    trace = []
    def check(step, cache):
        records = evaluate(job["task"], spec[1], val, lambda ids: decode(model, tok, cache, ids, cap))
        value = validation_nll(model, cache, val_labels)
        if not math.isfinite(value):
            raise ValueError(f"Non-finite validation NLL at step {step}")
        trace.append(dict(step=step, n=len(records), nll=value, **scores(job["task"], records)))
    if method == "icl":
        def predict(ids):
            return greedy_answer(model, tok, torch.tensor([direct_ids + ids], device=device), cap)
        cfg = model.config
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        cache_bytes = len(direct_ids) * (2 * cfg.num_hidden_layers * cfg.num_key_value_heads *
                                         head_dim * next(model.parameters()).element_size())
        meta.update(adaptation_s=0., adaptation_seconds=0., training_seconds=0., trainable_numel=0,
                    adaptation_peak_bytes=None, n_retained_demo_tokens=meta["n_demo_tokens"],
                    retained_cache_bytes=cache_bytes, cache_byte_source="analytic_full_prefix")
    else:
        eos = model.generation_config.eos_token_id
        eos_ids = eos if isinstance(eos, list) else [] if eos is None else [eos]
        newline_ids = tok.encode("\n", add_special_tokens=False) + tok.encode("\n\n", add_special_tokens=False)
        checkpoints = sorted({step for step in (0, 2, 4, 8, 16, job["updates"]) if step <= job["updates"]})
        cache, profile = fit(model, inst, demos, method=method, budget=job["budget"],
                             optimizer_seed=job["optimizer_seed"], selection_seed=job["selection_seed"],
                             steps=job["updates"], microbatch=2, eos_ids=eos_ids, newline_ids=newline_ids,
                             train_scope={"both":"kv", "key":"k", "value":"v"}[mode],
                             checkpoint_steps=checkpoints, on_checkpoint=check)
        if [r["step"] for r in trace] != checkpoints:
            raise ValueError("Missing validation checkpoint")
        meta.update(profile)
        meta.update(adaptation_s=profile["adaptation_seconds"], adaptation_peak_bytes=profile["adaptation_peak_allocated_bytes"],
                    cache_byte_source="measured_K_V_tensor_bytes")
        def predict(ids):
            return decode(model, tok, cache, ids, cap)
    write_json(out / "validation.json", trace)
    sync()
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    start = perf_counter()
    rows = evaluate(job["task"], spec[1], test, predict)
    sync()
    meta["inference_s"] = perf_counter() - start
    meta["inference_peak_bytes"] = torch.cuda.max_memory_allocated(device) if cuda else None
    meta["peak_allocated_bytes"] = max(meta["adaptation_peak_bytes"] or 0, meta["inference_peak_bytes"]) if cuda else None
    meta.update(scores(job["task"], rows), status="complete")
    write_atomically(out / "predictions.jsonl", "".join(json.dumps(r) + "\n" for r in rows).encode())
    write_json(out / "metrics.json", meta)
    print(f"{job['id']} completed: {meta['n_correct']}/{meta['n_scored']}", flush=True)


def main():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    job = json.loads(args.job.read_bytes())
    load_inputs(job)
    if args.model.name != MODEL_REV or not args.model.is_dir():
        parser.error(f"Expected a local {MODEL} Hugging Face snapshot at revision {MODEL_REV}.")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        parser.error("Use a CUDA device supporting bfloat16.")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
                                                 low_cpu_mem_usage=True, local_files_only=True).to("cuda")
    run_job(job, args.out, model, tok)


if __name__ == "__main__":
    main()
