"""Team reconstruction of the BBH setting in Lu et al. v3, Sections 5.1-5.3.

Public prompt strings and decode limits follow Fewshot-TTT ad40084 (src/tasks.py).
Demo/evaluation IDs come from the provided team manifest. Exact-match scoring is
case-sensitive after trimming outer whitespace on the first line. The authors'
BBH split, exact parser and checkpoint revision were not supplied in their release.
"""
import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from time import perf_counter

from bbh_eval.prepare_splits import (
    DEFAULT_MANIFEST, REVISION, TASK_HASHES, load_examples, write_atomically,
)

MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
MODEL_REV = "8afb486c1db24fe5011ec46dfbe5b5dccdb575c2"
PROMPT_SOURCE = "adamzweiger/Fewshot-TTT@ad40084c63f5cf503e59d2acdab426843dffd254"
LETTER = "Answer with only the corresponding letter (e.g. (A))."
# First sentence: original BBH instruction; second sentence/limits: Fewshot-TTT.
TASKS = {
    "boolean_expressions": ("Evaluate the result of a random Boolean expression.", "Answer with only 'True' or 'False'.", 1),
    "date_understanding": ("Infer the date from context.", LETTER, 3),
    "logical_deduction_three_objects": ("A logical deduction task which requires deducing the order of a sequence of objects.", LETTER, 3),
    "word_sorting": ("Sort a list of words.", "Answer with only a sequence of words.", 50),
}


def encode_instruction(tokenizer, task, spec=None):
    description, answer_format, _ = spec or TASKS[task]
    return [tokenizer.bos_token_id] + tokenizer.encode(f"{description} {answer_format}\n\n", add_special_tokens=False)


def encode_example(tokenizer, question, answer=None):
    prefix = f"Q: {question}\nA:"
    if answer is None:
        return tokenizer.encode(prefix, add_special_tokens=False), None, None
    target = f" {answer}"
    text = prefix + target + "\n\n"
    # A tokenizer may combine a label's closing parenthesis with a trailing
    # newline. Use offsets from a single encoding to retain exact prompt IDs.
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    answer_start = next(i for i, (_, end) in enumerate(offsets) if end > len(prefix))
    answer_end = next((i for i, (start, _) in enumerate(offsets) if start >= len(prefix + target)), len(offsets))
    return enc["input_ids"], answer_start, answer_end


@dataclass
class PreparedRun:
    instruction_ids: list
    demo_tokens: list
    questions: list
    meta: dict


def prepare(args, tokenizer, context_limit):
    raw = args.manifest.read_bytes()
    manifest = json.loads(raw)
    from bbh_eval.prepare_benchmark_splits import BENCHMARK_HASHES, BENCHMARK_PROTOCOL, SPEC_SHA, benchmark_rows, benchmark_split
    if args.task not in manifest["tasks"]:
        raise ValueError(f"{args.task} missing from {args.manifest}")
    cfg = manifest["tasks"][args.task]
    broad = manifest.get("protocol") == BENCHMARK_PROTOCOL
    if broad and args.split != "test":
        raise ValueError("Benchmark-wide rows use --split test.")
    available_hashes = BENCHMARK_HASHES if broad else TASK_HASHES
    if manifest["dataset_revision"] != REVISION or cfg["sha256"] != available_hashes[args.task]:
        raise ValueError("Manifest does not match the pinned BBH source.")
    if broad and manifest["task_spec_sha256"] != SPEC_SHA:
        raise ValueError("Unrecognized benchmark task specification.")
    rows = (benchmark_rows if broad else load_examples)(args.task, args.source_dir)
    spec = (cfg["task_prompt"], cfg["answer_format"], cfg["generation_length"]) if broad else TASKS[args.task]
    if args.max_new_tokens is None:
        args.max_new_tokens = spec[2]
    s = cfg["seeds"].get(str(args.seed))
    if s is None:
        raise ValueError(f"Seed {args.seed} missing from {args.manifest}")
    demo_ids = list(s["demo_ids"])
    if broad:
        dev_ids, test_ids = [], s["test_ids"]
        expected = benchmark_split(len(rows), args.seed, args.task, manifest.get("split_policy", "antecedent"))
        if (demo_ids, test_ids) != expected:
            raise ValueError("Benchmark IDs differ from the reconstructed shuffle.")
    else:
        dev_ids, test_ids = s["dev_ids"], cfg["test_ids"]
        if len(demo_ids) != 10 or sorted(demo_ids + dev_ids + test_ids) != list(range(len(rows))):
            raise ValueError("Expected 10 demonstrations and complete, disjoint splits.")
    available_ids = dev_ids if args.split == "dev" else test_ids
    eval_ids = available_ids[:args.limit]
    demos = [encode_example(tokenizer, rows[i]["input"], rows[i]["target"]) for i in demo_ids]
    lengths = [len(d[0]) for d in demos]
    all_demo_ids = list(demo_ids)
    selection_start = perf_counter()
    target = max(1, int(sum(lengths) * (args.whole_demo_budget or args.budget)))
    if args.whole_demo_budget is not None:
        from bbh_eval.whole_demo import choose_whole_demos
        keep = choose_whole_demos(lengths, target, args.selection_seed)
        if len(keep) < (2 if args.method == "full" else 1):
            raise ValueError("Too few complete demonstrations fit this token allowance.")
        demos, demo_ids = [demos[i] for i in keep], [demo_ids[i] for i in keep]
    selection_s = perf_counter() - selection_start if args.whole_demo_budget is not None else 0.
    instruction = encode_instruction(tokenizer, args.task, spec)
    prefix_ids = instruction + [t for demo, _, _ in demos for t in demo]
    description, fmt, _ = spec
    prefix_text = f"{description} {fmt}\n\n" + "".join(f"Q: {rows[i]['input']}\nA: {rows[i]['target']}\n\n" for i in demo_ids)
    questions = []
    for i in eval_ids:
        q = rows[i]["input"]
        ids = encode_example(tokenizer, q)[0]
        combined = [tokenizer.bos_token_id] + tokenizer.encode(prefix_text + f"Q: {q}\nA:", add_special_tokens=False)
        if combined != prefix_ids + ids:
            raise ValueError(f"Row {i}: segmentation changed the tokenizer input IDs.")
        if len(combined) + args.max_new_tokens > context_limit:
            raise ValueError(f"Row {i}: {len(combined)} + {args.max_new_tokens} exceeds context {context_limit}.")
        questions.append((i, q, ids, rows[i]["target"]))
    meta = dict(
        task=args.task, model=args.model, revision=args.revision, method=args.method, seed=args.seed,
        optimizer_seed=args.optimizer_seed, selection_seed=args.selection_seed, steps=args.steps,
        microbatch=args.microbatch, budget=args.budget, whole_demo_budget=args.whole_demo_budget,
        n_shots=len(demos), split=args.split, eligible_demo_ids=all_demo_ids, demo_ids=demo_ids,
        selected_demo_ids=demo_ids, dev_ids=dev_ids, evaluation_ids=eval_ids,
        requested_demo_tokens=target, n_demo_tokens=sum(lengths),
        n_selected_demo_tokens=sum(len(d[0]) for d in demos), n_instruction_tokens=len(instruction),
        whole_demo_selection_s=selection_s, max_new_tokens=args.max_new_tokens, context_limit=context_limit,
        max_input_tokens=max(len(prefix_ids) + len(q[2]) for q in questions),
        protocol=(BENCHMARK_PROTOCOL + "/" + manifest.get("split_policy", "antecedent") if broad else "team_four_fixed_test"),
        n_available_eval=len(available_ids), answer_format=fmt, dataset_revision=REVISION, dataset_sha256=cfg["sha256"], prompt_source=PROMPT_SOURCE,
        manifest_sha256=hashlib.sha256(raw).hexdigest(), prompt_protocol="bbh-direct-qa-v1",
        prompt_prefix_sha256=hashlib.sha256(prefix_text.encode()).hexdigest(),
    )
    return PreparedRun(instruction, demos, questions, meta)


def score(task, question, raw, gold):
    answer = raw.partition("\n")[0].strip()
    if task == "word_sorting":
        words = question.split("List:", 1)[1].split()
        valid = Counter(answer.split()) == Counter(words)
    elif task == "boolean_expressions":
        valid = answer in ("True", "False")
    elif re.search(r"^\s*\([A-Z]\)\s+\S", question, re.MULTILINE):
        valid = answer in re.findall(r"^\s*(\([A-Z]\))\s+\S", question, re.MULTILINE)
    else:
        valid = None
    return answer, valid, answer == gold


def antecedent_match(raw, gold, fmt):
    # Mirrors the cited prior baseline parser (Fewshot-TTT src/utils.py).
    answer = raw.strip().lower()
    if answer:
        if fmt in ("Answer with only a sequence of words.", "Answer with only a sequence of space-separated parentheses."):
            answer = answer.partition("\n")[0].partition("-")[0]
        else:
            answer = answer.split()[0]
        answer = answer.partition(".")[0].strip()
        if fmt == LETTER and answer and not answer.startswith("("):
            answer = f"({answer[0]})"
    return answer == gold.strip().lower()


def make_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", required=True, help="A task present in the supplied manifest.")
    p.add_argument("--model", choices=(MODEL, "meta-llama/Llama-3.2-3B-Instruct"), default=MODEL)
    p.add_argument("--method", choices=("icl", "full", "random", "answer_attention", "value_norm"), default="icl")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--source-dir", type=Path)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split", choices=("dev", "test"), default="dev")
    p.add_argument("--frozen", action="store_true", help="Use the test split after freezing the method.")
    p.add_argument("--limit", type=int, help="Optional small smoke test within the selected split.")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--revision", default=None, help="Full Hugging Face checkpoint commit SHA")
    p.add_argument("--budget", type=float, default=1., help="Fraction of demo tokens for entry selection.")
    p.add_argument("--whole-demo-budget", type=float, help="Whole-example control for icl/full only.")
    p.add_argument("--selection-seed", type=int, default=0)
    p.add_argument("--optimizer-seed", type=int, default=0)
    p.add_argument("--steps", type=int, default=16)
    p.add_argument("--microbatch", type=int, default=2)
    p.add_argument("--out", required=True, type=Path, help="Fresh run directory.")
    p.add_argument("--preflight", action="store_true", help="Load only config/tokenizer and validate all prompts.")
    return p


def greedy_answer(model, tokenizer, input_ids, max_new_tokens):
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList
    class Newline(StoppingCriteria):
        def __call__(self, ids, scores, **kwargs):
            text = tokenizer.decode(ids[0, input_ids.shape[1]:], skip_special_tokens=True)
            return "\n" in text
    with torch.inference_mode():
        ids = model.generate(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                             do_sample=False, num_beams=1, max_new_tokens=max_new_tokens,
                             pad_token_id=tokenizer.eos_token_id, use_cache=True,
                             stopping_criteria=StoppingCriteriaList([Newline()]))
    new = ids[0, input_ids.shape[1]:].tolist()
    return tokenizer.decode(new, skip_special_tokens=True, clean_up_tokenization_spaces=False), new


def main():
    p = make_parser()
    a = p.parse_args()
    if a.split == "test" and not a.frozen:
        p.error("Pass --frozen for the final held-out test after freezing settings.")
    if a.limit is not None and a.limit < 1:
        p.error("--limit must be positive.")
    if a.revision is None:
        if a.model != MODEL:
            p.error("Pass the full pinned SHA for the project model via --revision.")
        a.revision = MODEL_REV
    if (a.max_new_tokens is not None and a.max_new_tokens < 1) or not re.fullmatch(r"[0-9a-f]{40}", a.revision):
        p.error("Use a positive decode limit and a full model revision SHA.")
    if not 0 < a.budget <= 1 or (a.method in ("icl", "full") and a.budget != 1):
        p.error("--budget applies to random/answer_attention only and must be in (0,1].")
    if a.whole_demo_budget is not None and (a.method not in ("icl", "full") or not 0 < a.whole_demo_budget <= 1):
        p.error("--whole-demo-budget requires icl/full and a fraction in (0,1].")
    if a.microbatch < 1 or a.steps < 0:
        p.error("--microbatch must be positive and --steps nonnegative.")
    if a.out.exists():
        p.error(f"Choose a fresh output directory: {a.out} already exists.")

    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(a.model, revision=a.revision)
    tok = AutoTokenizer.from_pretrained(a.model, revision=a.revision, use_fast=True)
    run = prepare(a, tok, cfg.max_position_embeddings)
    m = run.meta
    m.update(torch=str(torch.__version__), transformers=transformers.__version__, cuda=torch.version.cuda,
             scoring="case-sensitive exact match of stripped first output line", attention_backend="sdpa",
             attention_score_backend="eager fallback when attention probabilities are requested")
    def save(status):
        m["status"] = status
        write_atomically(a.out / "metadata.json", (json.dumps(m, indent=2) + "\n").encode())
    a.out.mkdir(exist_ok=False, parents=True)
    if a.preflight:
        save("preflight_complete")
        print(f"{len(run.questions)} prompts fit. Longest prompt: {m['max_input_tokens']} tokens. No model weights loaded.")
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
    main()
