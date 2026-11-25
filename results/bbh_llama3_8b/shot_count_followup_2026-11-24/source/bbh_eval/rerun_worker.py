"""One BBH compression run using the original scoring, decoding, CT-KV and whole-demo selection."""
import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

from bbh_eval.prepare_benchmark_splits import BENCHMARK_HASHES, SPEC_SHA, benchmark_rows
from bbh_eval.prepare_splits import REVISION, write_atomically
from bbh_eval.rerun_data import PROTOCOL, SHOTS, make_split
from bbh_eval.run_llama_icl import (
    MODEL, MODEL_REV, PROMPT_SOURCE, PreparedRun, antecedent_match, encode_example,
    encode_instruction, greedy_answer, score,
)
from bbh_eval.whole_demo import choose_whole_demos


def worker_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--task", choices=sorted(BENCHMARK_HASHES), required=True)
    p.add_argument("--shots", type=int, choices=SHOTS, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--method", choices=("icl", "full", "random", "answer_attention", "value_norm"), required=True)
    p.add_argument("--budget", type=float, default=1.0, help="Fraction of demo KV positions for token selection.")
    p.add_argument("--whole-demo-budget", type=float)
    p.add_argument("--steps", type=int, choices=(0, 16), required=True)
    p.add_argument("--out", type=Path, required=True, help="Run-specific, fresh output directory.")
    p.add_argument("--selection-seed", type=int)
    p.add_argument("--optimizer-seed", type=int, default=0)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--limit", type=int, help="Debug only. Use a separate output root for limited evaluations.")
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--preflight", action="store_true", help="Check all context and support limits, without loading model weights.")
    p.add_argument("--model", choices=(MODEL,), default=MODEL)
    p.add_argument("--revision", choices=(MODEL_REV,), default=MODEL_REV)
    p.add_argument("--split", choices=("test",), default="test")
    p.add_argument("--frozen", action="store_true", help="Compatibility flag. The protocol fixes both candidates a priori.")
    return p


def validate_args(a):
    if a.method == "icl" and a.steps:
        raise ValueError("Use 0 updates for direct ICL.")
    if not 0 < a.budget <= 1 or (a.method in ("icl", "full") and a.budget != 1):
        raise ValueError("Token budgets below 1 require a token selection method.")
    if a.whole_demo_budget is not None and (a.method not in ("icl", "full") or not 0 < a.whole_demo_budget <= 1):
        raise ValueError("Whole-demo controls require icl/full and a budget in (0, 1].")
    if a.limit is not None and a.limit < 1 or a.microbatch <= 0:
        raise ValueError("Limit and microbatch must be positive.")
    if a.selection_seed is None:
        a.selection_seed = a.seed
    return a


def prepare_job(a, tokenizer, context_limit):
    """Check pinned source rows, fixed test IDs, and both inference and adaptation length."""
    payload = a.manifest.read_bytes()
    manifest = json.loads(payload)
    if (manifest["protocol"], manifest["dataset_revision"], manifest["task_spec_sha256"]) != (PROTOCOL, REVISION, SPEC_SHA):
        raise ValueError("Unrecognized rerun split, BBH revision or task-spec revision.")
    cfg = manifest["tasks"][a.task]
    if cfg["sha256"] != BENCHMARK_HASHES[a.task]:
        raise ValueError(f"Unexpected BBH digest for {a.task}.")
    rows = benchmark_rows(a.task, a.source_dir)
    expected = make_split(a.task, rows)
    seed = str(a.seed)
    if (cfg["test_ids"] != expected["test_ids"] or cfg["seeds"][seed] != expected["seeds"][seed]
            or cfg["support_pool_ids"] != expected["support_pool_ids"]):
        raise ValueError("Demo IDs, their order or fixed test IDs differ from the registered protocol.")
    eligible = cfg["seeds"][seed]["demo_ids"][:a.shots]
    evaluation_ids = cfg["test_ids"][:a.limit]
    spec = (cfg["task_prompt"], cfg["answer_format"], cfg["generation_length"])
    a.max_new_tokens = spec[2]
    inst = encode_instruction(tokenizer, a.task, spec)
    encoded = [encode_example(tokenizer, rows[i]["input"], rows[i]["target"]) for i in eligible]
    original_lengths = [len(d[0]) for d in encoded]
    n_original_tokens = sum(original_lengths)
    target_tokens = max(1, int(n_original_tokens * (a.whole_demo_budget or a.budget)))
    demo_ids = list(eligible)
    selection_s = 0.
    if a.whole_demo_budget is not None:
        start = perf_counter()
        picked = choose_whole_demos(original_lengths, target_tokens, a.selection_seed)
        demo_ids = [eligible[i] for i in picked]
        encoded = [encoded[i] for i in picked]
        selection_s = perf_counter() - start
    prefix_text = f"{spec[0]} {spec[1]}\n\n" + "".join(
        f"Q: {rows[i]['input']}\nA: {rows[i]['target']}\n\n" for i in demo_ids)
    prefix_ids = inst + [t for d in encoded for t in d[0]]
    m = dict(protocol=PROTOCOL, task=a.task, seed=a.seed, split="test", requested_shots=a.shots,
             n_shots=len(demo_ids), eligible_demo_ids=eligible, selected_demo_ids=demo_ids, demo_ids=demo_ids,
             evaluation_ids=evaluation_ids, n_available_eval=len(cfg["test_ids"]), dev_ids=[], method=a.method,
             steps=a.steps, budget=a.budget, whole_demo_budget=a.whole_demo_budget, selection_seed=a.selection_seed,
             optimizer_seed=a.optimizer_seed, microbatch=a.microbatch, whole_demo_selection_s=selection_s,
             n_demo_tokens=n_original_tokens, n_instruction_tokens=len(inst),
             n_selected_demo_tokens=len(prefix_ids)-len(inst), requested_demo_tokens=target_tokens,
             context_limit=context_limit, max_new_tokens=a.max_new_tokens, model=MODEL, revision=MODEL_REV,
             manifest_sha256=hashlib.sha256(payload).hexdigest(), dataset_revision=REVISION,
             dataset_sha256=BENCHMARK_HASHES[a.task], answer_format=spec[1], prompt_source=PROMPT_SOURCE,
             prompt_protocol="bbh-direct-qa-v1", prompt_prefix_sha256=hashlib.sha256(prefix_text.encode()).hexdigest())
    questions = []
    run = PreparedRun(inst, encoded, questions, m)
    if a.whole_demo_budget is not None and (not demo_ids or a.steps and len(demo_ids) < 2):
        m.update(status="infeasible", reason=("The original whole-demo ceiling fits no complete demonstration."
                 if not demo_ids else "The original leave-one-demonstration-out CT-KV requires at least two complete demonstrations."))
        return run
    if a.method != "icl" and a.microbatch > len(demo_ids):
        raise ValueError("The configured microbatch exceeds the actual demonstration count.")
    adapt_forward = a.steps > 0 or (a.method == "answer_attention" and a.budget < 1)
    m["max_adaptation_input_tokens"] = len(prefix_ids) + (max(e-1 for _, _, e in encoded) if adapt_forward else 0)
    if m["max_adaptation_input_tokens"] > context_limit:
        m.update(status="infeasible", reason=f"Adaptation or selection needs {m['max_adaptation_input_tokens']} positions, context limit {context_limit}.")
        return run
    longest = 0
    for i in evaluation_ids:
        q = rows[i]["input"]
        qids = encode_example(tokenizer, q)[0]
        total = len(prefix_ids) + len(qids)
        longest = max(longest, total)
        if total + a.max_new_tokens > context_limit:
            m.update(status="infeasible", max_input_tokens=longest,
                     reason=f"Test row {i} needs {total} input + {a.max_new_tokens} reserved output positions, context limit {context_limit}.")
            return run
        combined = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text + f"Q: {q}\nA:", add_special_tokens=False)
        if combined != prefix_ids + qids:
            raise ValueError(f"Row {i}: tokenizer changed IDs across the prompt boundary.")
        questions.append((i, q, qids, rows[i]["target"]))
    m["max_input_tokens"] = longest
    return run


def main():
    parser = worker_parser()
    a = parser.parse_args()
    try:
        validate_args(a)
    except ValueError as err:
        parser.error(str(err))
    if a.out.exists():
        parser.error(f"Choose a fresh output path; {a.out} already exists.")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, use_fast=True)
    run = prepare_job(a, tok, cfg.max_position_embeddings)
    m = run.meta
    m.update(torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
             python=platform.python_version(), scoring="case-sensitive exact match of stripped first output line",
             attention_backend="sdpa", attention_score_backend="eager fallback when attention probabilities are requested")
    def save(status):
        m["status"] = status
        write_atomically(a.out / "metadata.json", (json.dumps(m, indent=2) + "\n").encode())
    a.out.mkdir(exist_ok=False, parents=True)
    if m.get("status") == "infeasible":
        save("infeasible")
        print(f"Infeasible: {m['reason']}", flush=True)
        return
    if a.preflight:
        save("preflight_complete")
        print(f"{len(run.questions)} questions fit (longest input {m['max_input_tokens']}, selection/adaptation {m['max_adaptation_input_tokens']}).")
        return
    save("loading")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("Use a CUDA GPU supporting bfloat16 for this protocol.")
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True).to("cuda").eval()
    model.requires_grad_(False)
    m.update(gpu=torch.cuda.get_device_name(), dtype=str(next(model.parameters()).dtype))
    if a.method == "icl":
        prefix = run.instruction_ids + [t for demo, _, _ in run.demo_tokens for t in demo]
        m["adaptation_s"] = m["whole_demo_selection_s"]
        m["adaptation_peak_bytes"] = None
        m["n_retained_demo_tokens"] = m["n_selected_demo_tokens"]
        m["cache_byte_source"] = "analytic"
        head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
        bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * next(model.parameters()).element_size()
        m["retained_cache_bytes"] = len(prefix) * bytes_per_token
        m["full_logical_cache_bytes"] = (m["n_instruction_tokens"] + m["n_demo_tokens"]) * bytes_per_token
    else:
        from bbh_eval.ctkv import fit, decode
        eos = model.generation_config.eos_token_id
        eos_ids = [] if eos is None else eos if isinstance(eos, list) else [eos]
        nl = tok.encode("\n", add_special_tokens=False) + tok.encode("\n\n", add_special_tokens=False)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = perf_counter()
        prefix, extra = fit(model, run.instruction_ids, run.demo_tokens, a.method, a.budget,
                            a.optimizer_seed, a.selection_seed, a.steps, microbatch=a.microbatch,
                            eos_ids=eos_ids, newline_ids=nl)
        torch.cuda.synchronize()
        n_total = m["n_demo_tokens"]
        m.update(extra)
        m["n_demo_tokens"] = n_total
        m["cache_byte_source"] = "tensor_numel_x_element_size"
        m["full_logical_cache_bytes"] = (m["retained_cache_bytes"] // (m["n_instruction_tokens"] + m["n_retained_demo_tokens"])) * (m["n_instruction_tokens"] + n_total)
        m["adaptation_s"] = perf_counter() - start + m["whole_demo_selection_s"]
        m["adaptation_peak_bytes"] = torch.cuda.max_memory_allocated()
    save("evaluating")
    correct = invalid = comparable = prior_correct = 0
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = perf_counter()
    with (a.out / "predictions.jsonl").open("x") as f:
        for i, q, ids, gold in run.questions:
            if a.method == "icl":
                x = torch.tensor([prefix + ids], device="cuda")
                raw, new = greedy_answer(model, tok, x, a.max_new_tokens)
            else:
                raw, new = decode(model, tok, prefix, ids, a.max_new_tokens)
            pred, valid, right = score(a.task, q, raw, gold)
            prior_right = antecedent_match(raw, gold, m["answer_format"])
            row = dict(row_id=i, antecedent_exact_match=prior_right, gold=gold, prediction=pred, raw_generated_text=raw,
                       exact_match=right, valid_format=valid, generated_token_ids=new)
            f.write(json.dumps(row) + "\n")
            f.flush()
            correct += int(right)
            invalid += int(valid is False)
            comparable += int(valid is not None)
            prior_correct += int(prior_right)
    torch.cuda.synchronize()
    m.update(inference_s=perf_counter()-start, inference_peak_bytes=torch.cuda.max_memory_allocated(),
             n_scored=len(run.questions), n_correct=correct, n_invalid=invalid if comparable else None,
             n_validity_scored=comparable, antecedent_n_correct=prior_correct, accuracy=correct/len(run.questions))
    m["peak_allocated_bytes"] = max(m["adaptation_peak_bytes"] or 0, m["inference_peak_bytes"])
    save("complete")
    print(f"{a.method} {a.task} seed {a.seed}: {correct}/{len(run.questions)}, invalid {invalid}; {a.out}")


if __name__ == "__main__":
    from bbh_eval.rerun_errors import worker_main
    worker_main(main)
