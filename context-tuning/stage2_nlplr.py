"""Stage 2 (sparse demonstration-KV reading) runner for NLP-LR (GPT-2 Large).

Same trajectory as study_nlplr.run_job (prefill, leave-one-out + 5% token dropout, full-batch AdamW 1e-3, cosine
schedule over `epochs`, raw token-mean loss, option scoring), with two additions:
  adapt  {"mode": prod|dense|topk|fixed|random, "r": float}  how the KV prefix is read inside every training forward
         (prod = production SDPA kernel, i.e. the Stage 1 path; dense = the diagnostic implementation with r = 1)
  reads  [{"name", "mode", "r"}]  each checkpoint is scored once per read (same checkpoint, same items)
Other fields: as study_nlplr (id, out, task, demos, loss, epochs, checkpoints, seed, meta) plus mask_seed (random
rule; training reseeds it to mask_seed, every evaluation read reseeds it to mask_seed + 1).
fixed = Ryan's answer_attention ranking, computed once on the initial (dense) prefix: for each demonstration, mask
its own prefix span, run it from the cache, average the attention its answer-predicting positions put on every prefix
position over layers and heads; positions are scored by the mean over demonstrations (equal weight per demo).

Usage (from context-tuning/): .venv/bin/python stage2_nlplr.py JOBS.jsonl [--only PREFIX]
"""
import argparse
import gc
import json
import os
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

import ctkv_sparse as SP
from ctkv_perf import PhaseProfiler, tensor_bytes
from data_utils import GSDataset, collate_fn_gs
from study_nlplr import LR, TOKEN_DROPOUT, build_eval, evaluate, load_tests, set_all_seeds


def train_batch_tensors(batch, starts, kv_len, device, dropout):
    ids = batch["input_ids"].to(device)
    amask = batch["attention_mask"].to(device)
    labels = batch["label_ids"].to(device)
    bs = ids.shape[0]
    kv_mask = torch.ones((bs, kv_len), device=device, dtype=torch.int64)
    for b, idx in enumerate(batch["example_idx"]):
        s = starts[idx]
        e = starts[idx + 1] if idx < len(starts) - 1 else kv_len
        kv_mask[b, s:e] = 0
    if dropout:
        drop = (torch.rand_like(kv_mask, dtype=torch.float) > dropout).float()
        kv_mask = (kv_mask * drop).long()
    return ids, amask, labels, kv_mask


def forward_from_cache(model, kv, prefix_len, ids, amask, kv_mask, labels=None, attentions=False):
    bs = ids.shape[0]
    bkv = tuple((k.expand(bs, -1, -1, -1), v.expand(bs, -1, -1, -1)) for k, v in kv)
    pos = torch.zeros((bs, ids.shape[1]), device=ids.device, dtype=torch.int64)
    for row, n in zip(pos, amask.sum(1)):
        row[:n] = torch.arange(prefix_len, prefix_len + n, device=ids.device)
    embeds = model.transformer.wte(ids)
    full_mask = torch.cat([kv_mask, amask], dim=1)
    return model(inputs_embeds=embeds, attention_mask=full_mask, labels=labels, use_cache=True,
                 past_key_values=bkv, position_ids=pos, output_attentions=attentions)


@torch.no_grad()
def answer_attention(model, kv, prefix_len, gs, starts, device):
    """Ryan's answer_attention rule for GPT-2: mean attention of answer-predicting positions on prefix positions."""
    loader = DataLoader(gs, batch_size=len(gs), shuffle=False, collate_fn=partial(collate_fn_gs, dataset=gs))
    total = torch.zeros(prefix_len, device=device)
    n_demos = 0
    for batch in loader:
        ids, amask, labels, kv_mask = train_batch_tensors(batch, starts, prefix_len, device, 0.0)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = forward_from_cache(model, kv, prefix_len, ids, amask, kv_mask, attentions=True)
        # position t predicts token t+1; answer-predicting positions are those whose next label is not -100
        pred = torch.zeros_like(labels, dtype=torch.bool)
        pred[:, :-1] = labels[:, 1:] != -100
        for b in range(ids.shape[0]):
            q = pred[b].nonzero().flatten()
            acc = torch.zeros(prefix_len, device=device)
            for layer in out.attentions:
                acc += layer[b][:, q, :prefix_len].float().mean(dim=(0, 1))
            total += acc / len(out.attentions)
            n_demos += 1
        del out
    return total / max(n_demos, 1)


def run_job(job, model, tokenizer):
    set_all_seeds(job.get("seed", 0))
    task, demos = job["task"], job["demos"]
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    epochs = job.get("epochs", 200)
    checkpoints = sorted(set(job.get("checkpoints", [0, epochs])))
    stop_at = job.get("stop_at") or epochs
    adapt = job.get("adapt", {"mode": "prod", "r": 1.0})
    reads = job.get("reads", [{"name": "prod", "mode": "prod", "r": 1.0}])
    mask_seed = job.get("mask_seed", 0)
    if job.get("loss", "raw") != "raw":
        raise ValueError("Stage 2 uses the raw loss only")
    res = dict(id=job["id"], task=task, loss="raw", epochs=epochs, stop_at=stop_at, meta=job.get("meta", {}),
               adapt=adapt, reads=reads, mask_seed=mask_seed,
               demo_pool_ids=[d.get("pool_id") for d in demos], demo_labels=[d["output"] for d in demos],
               checkpoints={}, timing={}, sparse_stats={})

    t0 = time.perf_counter()
    tests = load_tests(task)
    prefix_ids, starts, items = build_eval(tokenizer, demos, tests)
    res["n_test"] = len({x["test_id"] for x in items})
    res["prefix_len"] = P = int(prefix_ids.shape[0])
    res["timing"]["build_s"] = time.perf_counter() - t0
    owner = torch.full((P,), -1, dtype=torch.long)
    for i, s in enumerate(starts):
        e = starts[i + 1] if i < len(starts) - 1 else P
        owner[s:e] = i
    res["n_prefix_demo_tokens"] = int((owner >= 0).sum())
    res["n_prefix_other_tokens"] = int((owner < 0).sum())

    prof = PhaseProfiler()
    prof.note(model_weights=tensor_bytes(list(model.parameters()) + list(model.buffers())),
              resident_before_job=torch.cuda.memory_allocated())
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_adapt = time.perf_counter()
    with prof("prefill"):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            kv = model(input_ids=prefix_ids.unsqueeze(0).to(device)).past_key_values
        kv = tuple((k.detach().to(torch.bfloat16).clone(), v.detach().to(torch.bfloat16).clone()) for k, v in kv)
    prof.note(kv_cache=tensor_bytes([x for layer in kv for x in layer]), prefix_tokens=P)
    gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(demos)}, tokenizer=tokenizer, pad_side="right",
                   past_kv_len=P)
    res["effective_shots"] = len(gs)
    fixed = None
    if adapt["mode"] == "fixed" or any(r["mode"] == "fixed" for r in reads):
        with prof("select_init"):
            fixed = answer_attention(model, kv, P, gs, starts, device)
        res["fixed_score_top"] = fixed.argsort(descending=True)[:32].tolist()
    # NLP-LR prefixes are demonstrations only (no instruction); every prefix position is selectable.
    SP.configure(P, owner >= 0, owner=owner, n_demos=len(starts), fixed_score=fixed, mask_seed=mask_seed)
    eval_time = 0.0

    def checkpoint(step):
        nonlocal eval_time
        torch.cuda.synchronize()
        s = time.perf_counter()
        with prof("eval"):
            for rd in reads:
                SP.reseed(mask_seed + 1)
                SP.STATE.reset_stats()
                torch.cuda.synchronize()
                t = time.perf_counter()
                with SP.sparse(rd["mode"], rd["r"]):
                    score, rows = evaluate(model, kv, P, items, pad_id, task)
                torch.cuda.synchronize()
                res["checkpoints"][f"{step}|{rd['name']}"] = dict(
                    step=step, read=rd["name"], score=score, predictions=rows, eval_s=time.perf_counter() - t,
                    n_items=len(items), sparse=SP.STATE.stats() if rd["mode"] not in ("prod", "off") else None)
        torch.cuda.synchronize()
        eval_time += time.perf_counter() - s

    if 0 in checkpoints:
        checkpoint(0)
    losses = []
    if stop_at > 0 and epochs > 0:
        params = [x for layer in kv for x in layer]
        for p in params:
            p.requires_grad_(True)
        loader = DataLoader(gs, batch_size=len(gs), shuffle=True, collate_fn=partial(collate_fn_gs, dataset=gs),
                            drop_last=False, num_workers=0)
        optim = torch.optim.AdamW([{"params": params, "lr": LR}], weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)
        model.train()
        SP.reseed(mask_seed)
        SP.STATE.reset_stats()
        train_stats = None
        steps_done = 0
        for step in range(1, stop_at + 1):
            t_step = time.perf_counter()
            step_loss = 0.0
            for batch in loader:
                with prof("prep"):
                    ids, amask, labels, kv_mask = train_batch_tensors(batch, starts, P, device, TOKEN_DROPOUT)
                with prof("forward"):
                    with torch.autocast("cuda", dtype=torch.bfloat16), SP.sparse(adapt["mode"], adapt["r"]):
                        out = forward_from_cache(model, kv, P, ids, amask, kv_mask, labels=labels)
                        loss = out.loss * ids.shape[0] / len(gs)
                with prof("backward"):
                    loss.backward()
                step_loss += loss.item()
                del out
            losses.append(step_loss)
            with prof("clip"):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
            if step == 1:
                prof.note(grads=tensor_bytes([p.grad for p in params]))
            with prof("optimizer"):
                optim.step()
                sched.step()
                optim.zero_grad()
            if step == 1:
                prof.note(optimizer_state=tensor_bytes([v for st in optim.state.values() for v in st.values()
                                                        if torch.is_tensor(v) and v.is_cuda]))
            torch.cuda.synchronize()
            prof.step_s.append(time.perf_counter() - t_step)
            steps_done = step
            if step in checkpoints:
                if adapt["mode"] not in ("prod", "off"):
                    train_stats = SP.STATE.stats()
                model.eval()
                for p in params:
                    p.requires_grad_(False)
                checkpoint(step)
                for p in params:
                    p.requires_grad_(True)
                model.train()
        model.eval()
        for p in params:
            p.requires_grad_(False)
        del optim, sched
        res["optimizer_steps"] = steps_done
        res["sparse_stats"]["train"] = train_stats
    torch.cuda.synchronize()
    res["timing"]["adapt_s"] = time.perf_counter() - t_adapt - eval_time
    res["timing"]["checkpoint_eval_s"] = eval_time
    res["timing"]["peak_mem_mb"] = max(prof.overall_peak, torch.cuda.max_memory_allocated()) / 2**20
    res["perf"] = prof.summary(res["timing"]["adapt_s"])
    res["train_losses"] = losses
    del kv
    gc.collect()
    torch.cuda.empty_cache()
    return res


def load_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", cache_dir="./.cache")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("openai-community/gpt2-large", attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
                                        summary_first_dropout=0.0, _attn_implementation="sdpa")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large", config=config, torch_dtype=torch.bfloat16,
                                            cache_dir="./.cache").cuda().eval()
    model.requires_grad_(False)
    SP.install()
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.only:
        jobs = [j for j in jobs if j["id"].startswith(a.only)]
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    if not todo:
        return
    model, tokenizer = load_model()
    for n, job in enumerate(todo):
        if os.path.exists(job["out"]):
            continue
        t = time.time()
        res = run_job(job, model, tokenizer)
        res["wall_s"] = time.time() - t
        res["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=sys.version.split()[0])
        os.makedirs(os.path.dirname(job["out"]) or ".", exist_ok=True)
        tmp = job["out"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, job["out"])
        scores = {k: round(v["score"] * 100, 2) for k, v in res["checkpoints"].items()}
        print(f"[{n + 1}/{len(todo)}] {job['id']} n_test={res['n_test']} {scores} adapt={res['timing']['adapt_s']:.1f}s "
              f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
