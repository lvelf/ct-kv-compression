"""Evaluate the pinned upstream KVzip physical eviction engine on the team's BBH inputs."""
import json
import re
import subprocess
import tempfile
import time
from pathlib import Path

from bbh_eval.run_llama_icl import MODEL, MODEL_REV, TASKS, make_parser, prepare, score
from bbh_eval.prepare_splits import write_atomically

UPSTREAM = "5d8472975d286e3da6b52178af39d18904330c0d"


def bytes_used(kv):
    return sum(t.numel() * t.element_size()
               for group in (kv.key_cache, kv.value_cache) for t in group)


def matched_random(kv, counts, seed, torch):
    gen = torch.Generator().manual_seed(seed)
    if len(counts) != kv.n_layers or any(len(row) != kv.n_heads_kv for row in counts):
        raise ValueError("Reference has incompatible layer/head dimensions.")
    valid = torch.zeros(kv.n_layers, 1, kv.n_heads_kv, kv.ctx_len, dtype=torch.bool)
    for layer, heads in enumerate(counts):
        for head, keep in enumerate(heads):
            if type(keep) is not int or not 0 <= keep <= kv.ctx_len:
                raise ValueError("Invalid reference head budget.")
            valid[layer, 0, head, torch.randperm(kv.ctx_len, generator=gen)[:keep]] = True
    kv.valid = valid.to(kv.device)
    kv.prepare_init()
    kv.pruned = True


def generate_one(w, kv, query, limit, torch):
    # Native wrapper assigns positions from the absolute length recorded by its cache.
    # Retain the last token if the model reaches the cap without emitting EOS.
    start = kv.get_seq_length()
    ids = w.model.generation_config.eos_token_id
    endings = {w.tokenizer.eos_token_id, *(ids if isinstance(ids, list) else [ids])}
    generated = []
    tokens = torch.tensor([query], device=w.device)
    with torch.inference_mode():
        for _ in range(limit):
            hidden = w.model.model(input_ids=tokens, past_key_values=kv, use_cache=True).last_hidden_state
            token = w.model.lm_head(hidden[:, -1:]).argmax(-1).item()
            generated.append(token)
            text = w.tokenizer.decode(generated, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            if token in endings or "\n" in text:
                break
            tokens = torch.tensor([[token]], device=w.device)
    kv.slice(start)
    return text, generated


def main():
    p = make_parser()
    p.description = __doc__
    p.set_defaults(steps=0)
    p.add_argument("--random-shape-from", type=Path,
                   help="Native KVzip metadata for the identical prefix and realized layer/head counts.")
    args = p.parse_args()
    if args.method != "icl" or args.whole_demo_budget is not None or args.preflight:
        p.error("This entry point takes --budget, with optional --random-shape-from. Omit --method.")
    if not (0 < args.budget <= 1) or (args.limit is not None and args.limit < 1):
        p.error("Expected a positive limit and a budget in (0,1].")
    if args.split == "test" and not args.frozen:
        p.error("Use --frozen after fixing the test protocol.")
    if args.steps != 0:
        p.error("Native KVzip has no gradient updates. Use --steps 0.")
    if args.task not in TASKS:
        p.error("This project comparator covers the four proposal tasks.")
    if args.revision is None:
        if args.model != MODEL:
            p.error("Supply the project model revision explicitly.")
        args.revision = MODEL_REV
    if (args.max_new_tokens is not None and args.max_new_tokens <= 0) or not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        p.error("Expected a positive token cap and full model revision SHA.")
    if args.out.exists():
        p.error("Choose a fresh output path.")
    if args.random_shape_from and args.budget == 1:
        p.error("At full budget the two controls are identical.")

    import torch
    import transformers
    from huggingface_hub import snapshot_download
    from model import ModelKVzip
    import model.wrapper as wrapper_module

    upstream = Path(wrapper_module.__file__).resolve().parents[1]
    rev = subprocess.check_output(["git", "-C", str(upstream), "rev-parse", "HEAD"], text=True).strip()
    if rev != UPSTREAM:
        p.error(f"Use the pinned upstream KVzip commit {UPSTREAM} (got {rev}).")
    model_id = getattr(args, "model", MODEL)
    snapshot = snapshot_download(repo_id=model_id, revision=args.revision,
                                 allow_patterns=["*.safetensors", "*.json", "*.model", "*.txt"])
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("Expose one CUDA GPU to this process (CUDA_VISIBLE_DEVICES).")
    with tempfile.TemporaryDirectory() as links:
        path = Path(links) / model_id.split("/")[-1]
        path.symlink_to(snapshot, target_is_directory=True)
        w = ModelKVzip(str(path), kv_type="evict")
    if next(w.model.parameters()).dtype != torch.bfloat16:
        raise ValueError("Expected the reference checkpoint to load in bfloat16.")
    run = prepare(args, w.tokenizer, w.config.max_position_embeddings)
    meta = run.meta
    meta.update(method="kvzip_native_random" if args.random_shape_from else "kvzip_native",
                steps=0, model=model_id, revision=args.revision, source_commit=rev,
                torch=str(torch.__version__), transformers=transformers.__version__,
                cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(), dtype=str(w.model.dtype),
                attention_backend="upstream FlashAttention 2", native_evict=True)
    ref = None
    if args.random_shape_from:
        ref = json.loads(args.random_shape_from.read_text())
        same = ["task", "seed", "model", "revision", "manifest_sha256", "prompt_prefix_sha256",
                "demo_ids", "budget", "source_commit"]
        if (ref.get("status") != "complete" or ref.get("method") != "kvzip_native"
                or any(ref.get(k) != meta.get(k) for k in same)):
            raise ValueError("Random control and native reference use different conditions.")

    args.out.mkdir(parents=True, exist_ok=False)
    def save(status):
        meta["status"] = status
        write_atomically(args.out / "metadata.json", (json.dumps(meta, indent=2) + "\n").encode())
    def clock():
        torch.cuda.synchronize()
        return time.perf_counter()

    save("evaluating")
    w.sys_prompt_ids = torch.tensor([run.instruction_ids], device=w.device)
    demo = torch.tensor([[t for tokens, _, _ in run.demo_tokens for t in tokens]], device=w.device)
    torch.cuda.reset_peak_memory_stats()
    start = clock()
    kv = w.prefill(demo, do_score=False)
    scored_from = clock()
    meta["uncompressed_cache_bytes"] = bytes_used(kv)
    if ref is None:
        w.scoring(kv, demo, load_score=False)
        _, actual = kv.prune(ratio=args.budget, level="pair")
    else:
        matched_random(kv, ref["retained_pairs_by_layer_head"], args.selection_seed, torch)
        actual = float(kv.valid.float().mean())
    end = clock()
    counts = kv.valid.sum(-1).squeeze(1).cpu().tolist()
    meta.update(retained_cache_bytes=bytes_used(kv), realized_demo_fraction=actual,
                retained_pairs_by_layer_head=counts,
                n_retained_demo_pairs=sum(sum(row) for row in counts),
                prefix_init_seconds=scored_from-start, selection_seconds=end-scored_from,
                adaptation_s=end-start, adaptation_peak_bytes=torch.cuda.max_memory_allocated())
    if ref is not None and meta["retained_cache_bytes"] != ref["retained_cache_bytes"]:
        raise ValueError("Random control cache size differs from native KVzip.")
    if hasattr(kv, "score"):
        del kv.score
    torch.cuda.reset_peak_memory_stats()
    start = clock()
    n_correct = n_invalid = 0
    with (args.out / "predictions.jsonl").open("x") as f:
        for i, question, ids, gold in run.questions:
            raw, generated = generate_one(w, kv, ids, args.max_new_tokens, torch)
            pred, valid, right = score(args.task, question, raw, gold)
            row = dict(row_id=i, prediction=pred, gold=gold, valid_format=valid,
                       exact_match=right, raw_generated_text=raw, generated_token_ids=generated)
            print(json.dumps(row), file=f, flush=True)
            n_correct += int(right)
            n_invalid += int(not valid)
    meta.update(inference_s=clock()-start, inference_peak_bytes=torch.cuda.max_memory_allocated(),
                n_scored=len(run.questions), n_correct=n_correct, n_invalid=n_invalid,
                accuracy=n_correct/len(run.questions))
    meta["peak_allocated_bytes"] = max(meta["adaptation_peak_bytes"], meta["inference_peak_bytes"])
    save("complete")
    print(f"{meta['method']}: {n_correct}/{len(run.questions)}, {meta['retained_cache_bytes']} KV bytes")


if __name__ == "__main__":
    main()
