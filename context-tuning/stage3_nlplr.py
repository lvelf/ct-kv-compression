"""Stage 3 round 1 runner for NLP-LR (GPT-2 Large): half-delta KV update and same-trajectory early stopping.

Guideline: docs/experiment-guideline/CTKV_Stage3_Round1_Guideline.md. One production-dense CT-KV trajectory per
(task, demonstration seed), exactly study_nlplr.run_job with the raw loss (prefill, leave-one-out + 5% token
dropout, full-batch AdamW 1e-3, cosine schedule over T = epochs = 200, option scoring, production SDPA), scored
under five conditions:
  icl            C_0 (step 0)
  early_quarter  C_{T/4} of the real trajectory (step 50; the LR schedule still spans T)
  early_half     C_{T/2} (step 100)
  full           C_T (step 200)
  half_delta     C_0 + 0.5 (C_T - C_0), per layer K and V: torch.lerp(C_0.float(), C_T.float(), 0.5).to(bf16)
C_0 is a detached clone taken right after prefill. Mid-trajectory evaluations save and restore the python / numpy /
torch CPU / CUDA RNG states and the model's train flag, so they cannot change later dropout or batch order.
Every job also records the displacement geometry (norm ratio and cosine of C_t - C_0 against C_T - C_0, K and V
separately). Jobs with "smoke": true also check the interpolation code path at weight 0 and 1 (bitwise endpoints
and identical predictions), that C_0 and the training cache are not modified by evaluation, and question alignment.

Usage (from context-tuning/): .venv/bin/python stage3_nlplr.py JOBS.jsonl [--only PREFIX]
"""
import argparse
import gc
import json
import os
import random
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from ctkv_perf import PhaseProfiler, tensor_bytes
from data_utils import GSDataset, collate_fn_gs
from study_nlplr import LR, TOKEN_DROPOUT, build_eval, evaluate, load_tests, set_all_seeds

CONDITIONS = ("icl", "early_quarter", "early_half", "full", "half_delta")


def rng_state():
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())


def set_rng_state(s):
    random.setstate(s[0])
    np.random.set_state(s[1])
    torch.set_rng_state(s[2])
    torch.cuda.set_rng_state_all(s[3])


def clone_cache(kv):
    return tuple((k.detach().clone(), v.detach().clone()) for k, v in kv)


def interp(c0, ct, w):
    """C_0 + w (C_T - C_0) per layer, computed in float32 and cast back to the cache dtype."""
    with torch.no_grad():
        return tuple((torch.lerp(k0.float(), k1.float(), w).to(k1.dtype), torch.lerp(v0.float(), v1.float(), w).to(v1.dtype))
                     for (k0, v0), (k1, v1) in zip(c0, ct))


def checksum(kv):
    """Exact integer checksum of the bf16 bit patterns (plain and position-weighted), K and V of every layer."""
    with torch.no_grad():
        out = []
        for layer in kv:
            for x in layer:
                b = x.contiguous().view(torch.int16).flatten().long()
                w = torch.arange(1, b.numel() + 1, device=b.device) % 65521
                out += [int(b.sum()), int((b * w).sum())]
        return out


def bitwise_equal(a, b):
    return all(torch.equal(x, y) for la, lb in zip(a, b) for x, y in zip(la, lb))


def geometry(c0, ct, cT):
    """Norm ratio |C_t - C_0| / |C_T - C_0| and cosine between the two displacements, K and V separately."""
    out = {}
    with torch.no_grad():
        for j, name in ((0, "K"), (1, "V")):
            dt = torch.cat([(l1[j].float() - l0[j].float()).flatten() for l0, l1 in zip(c0, ct)])
            dT = torch.cat([(l1[j].float() - l0[j].float()).flatten() for l0, l1 in zip(c0, cT)])
            out[name] = dict(norm_t=float(dt.norm()), norm_T=float(dT.norm()),
                             norm_ratio=float(dt.norm() / dT.norm()) if dT.norm() > 0 else float("nan"),
                             cosine=float(torch.nn.functional.cosine_similarity(dt, dT, dim=0)),
                             rel_to_C0=float(dT.norm() / torch.cat([l0[j].float().flatten() for l0 in c0]).norm()))
    return out


def run_job(job, model, tokenizer):
    set_all_seeds(job.get("seed", 0))
    task, demos = job["task"], job["demos"]
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    T = job.get("epochs", 200)
    steps = {"icl": 0, "early_quarter": T // 4, "early_half": T // 2, "full": T}
    smoke = bool(job.get("smoke"))
    if job.get("loss", "raw") != "raw":
        raise ValueError("Stage 3 round 1 uses the raw loss only")
    res = dict(id=job["id"], task=task, loss="raw", epochs=T, condition_steps=steps, meta=job.get("meta", {}),
               stage1_id=job.get("stage1_id"), demo_pool_ids=[d.get("pool_id") for d in demos],
               demo_labels=[d["output"] for d in demos], conditions={}, timing={}, checks={},
               scheduler=dict(type="cosine_with_warmup", warmup=0, num_training_steps=T, lr=LR, optimizer="AdamW",
                              weight_decay=0.0, clip=1.0, token_dropout=TOKEN_DROPOUT, batch="full"))

    t0 = time.perf_counter()
    tests = load_tests(task)
    prefix_ids, starts, items = build_eval(tokenizer, demos, tests)
    P = int(prefix_ids.shape[0])
    res["n_test"] = len({x["test_id"] for x in items})
    res["prefix_len"] = P
    res["timing"]["build_s"] = time.perf_counter() - t0

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
    torch.cuda.synchronize()
    res["timing"]["prefill_s"] = time.perf_counter() - t_adapt
    c0 = clone_cache(kv)                     # independent, read-only C_0
    c0_sum = checksum(c0)
    prof.note(kv_cache=tensor_bytes([x for layer in kv for x in layer]), prefix_tokens=P)
    eval_time = 0.0
    snaps = {}

    def score(cond, cache, step):
        nonlocal eval_time
        torch.cuda.synchronize()
        s = time.perf_counter()
        with prof("eval"):
            sc, rows = evaluate(model, cache, P, items, pad_id, task)
        torch.cuda.synchronize()
        dt = time.perf_counter() - s
        eval_time += dt
        acc = float(np.mean([r[1] == r[2] for r in rows])) if rows else float("nan")
        res["conditions"][cond] = dict(condition=cond, step=step, score=sc, accuracy=acc, predictions=rows,
                                       n_questions=len(rows), eval_s=dt)
        return rows

    def adapt_so_far():
        torch.cuda.synchronize()
        return time.perf_counter() - t_adapt - eval_time

    res["timing"]["cumulative_adapt_s"] = {"icl": adapt_so_far()}
    score("icl", kv, 0)
    losses = []
    params = [x for layer in kv for x in layer]
    for p in params:
        p.requires_grad_(True)
    gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(demos)}, tokenizer=tokenizer, pad_side="right",
                   past_kv_len=P)
    batch_size = len(gs)
    loader = DataLoader(gs, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_gs, dataset=gs),
                        drop_last=False, num_workers=0)
    optim = torch.optim.AdamW([{"params": params, "lr": LR}], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=T)
    model.train()
    kv_len = kv[0][0].shape[2]
    n_updates = 0
    for step in range(1, T + 1):
        t_step = time.perf_counter()
        step_loss = 0.0
        for batch in loader:
            with prof("prep"):
                ids = batch["input_ids"].to(device)
                amask = batch["attention_mask"].to(device)
                labels = batch["label_ids"].to(device)
                bs = ids.shape[0]
                kv_mask = torch.ones((bs, kv_len), device=device, dtype=torch.int64)
                bkv = tuple((k.expand(bs, -1, -1, -1), v.expand(bs, -1, -1, -1)) for k, v in kv)
                for b, idx in enumerate(batch["example_idx"]):
                    s = starts[idx]
                    e = starts[idx + 1] if idx < len(starts) - 1 else kv_len
                    kv_mask[b, s:e] = 0
                if TOKEN_DROPOUT:
                    drop = (torch.rand_like(kv_mask, dtype=torch.float) > TOKEN_DROPOUT).float()
                    kv_mask = (kv_mask * drop).long()
            with prof("forward"):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pos = torch.zeros((bs, ids.shape[1]), device=device, dtype=torch.int64)
                    for row, n in zip(pos, amask.sum(1)):
                        row[:n] = torch.arange(P, P + n, device=device)
                    embeds = model.transformer.wte(ids)
                    full_mask = torch.cat([kv_mask, amask], dim=1)
                    out = model(inputs_embeds=embeds, attention_mask=full_mask, labels=labels, use_cache=True,
                                past_key_values=bkv, position_ids=pos)
                    loss = out.loss * bs / batch_size
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
        n_updates += 1
        if step == 1:
            prof.note(optimizer_state=tensor_bytes([v for st in optim.state.values() for v in st.values()
                                                    if torch.is_tensor(v) and v.is_cuda]))
        torch.cuda.synchronize()
        prof.step_s.append(time.perf_counter() - t_step)
        for cond in ("early_quarter", "early_half", "full"):
            if step == steps[cond]:
                res["timing"]["cumulative_adapt_s"][cond] = adapt_so_far()
                res.setdefault("optimizer_updates_at", {})[cond] = n_updates
                st, was_training = rng_state(), model.training
                model.eval()
                for p in params:
                    p.requires_grad_(False)
                if cond != "full":
                    snaps[cond] = clone_cache(kv)
                before = checksum(kv) if smoke else None
                score(cond, kv, step)
                if smoke:
                    res["checks"][f"train_cache_unchanged_by_eval@{cond}"] = checksum(kv) == before
                for p in params:
                    p.requires_grad_(True)
                model.train(was_training)
                set_rng_state(st)
    model.eval()
    for p in params:
        p.requires_grad_(False)
    res["optimizer_steps"] = n_updates
    res["scheduler"]["last_lr"] = sched.get_last_lr()[0]
    res["scheduler"]["scheduler_steps"] = sched.last_epoch
    del optim, sched
    # half_delta: interpolate after the full trajectory, evaluate on an independent cache
    torch.cuda.synchronize()
    s = time.perf_counter()
    with prof("interp"):
        half = interp(c0, kv, 0.5)
    torch.cuda.synchronize()
    res["timing"]["interp_s"] = time.perf_counter() - s
    score("half_delta", half, T)
    res["timing"]["cumulative_adapt_s"]["half_delta"] = res["timing"]["cumulative_adapt_s"]["full"] + \
        res["timing"]["interp_s"]
    res["geometry"] = {c: geometry(c0, snaps[c], kv) for c in ("early_quarter", "early_half")}
    res["geometry"]["half_delta"] = geometry(c0, half, kv)
    if smoke:
        w0, w1 = interp(c0, kv, 0.0), interp(c0, kv, 1.0)
        res["checks"]["interp_w0_bitwise_C0"] = bitwise_equal(w0, c0)
        res["checks"]["interp_w1_bitwise_CT"] = bitwise_equal(w1, kv)
        for name, cache, ref in (("w0", w0, "icl"), ("w1", w1, "full")):
            _, rows = evaluate(model, cache, P, items, pad_id, task)
            refrows = res["conditions"][ref]["predictions"]
            res["checks"][f"interp_{name}_same_predictions_as_{ref}"] = [r[:3] for r in rows] == [r[:3] for r in refrows]
            res["checks"][f"interp_{name}_max_abs_option_loss_diff_vs_{ref}"] = max(
                abs(a - b) for r, q in zip(rows, refrows) for a, b in zip(r[3], q[3]))
        del w0, w1
    res["checks"]["C0_unchanged"] = checksum(c0) == c0_sum
    ids_by = {c: [r[0] for r in v["predictions"]] for c, v in res["conditions"].items()}
    res["checks"]["question_ids_aligned"] = all(v == ids_by["icl"] for v in ids_by.values())
    res["checks"]["conditions_present"] = sorted(res["conditions"]) == sorted(CONDITIONS)
    torch.cuda.synchronize()
    res["timing"]["adapt_s"] = res["timing"]["cumulative_adapt_s"]["full"]
    res["timing"]["eval_s_total"] = eval_time
    res["timing"]["peak_mem_mb"] = max(prof.overall_peak, torch.cuda.max_memory_allocated()) / 2**20
    res["perf"] = prof.summary(res["timing"]["adapt_s"])
    res["train_losses"] = losses
    del kv, c0, half, snaps
    gc.collect()
    torch.cuda.empty_cache()
    return res


def validate(res, job):
    assert sorted(res["conditions"]) == sorted(CONDITIONS), res["conditions"].keys()
    n = {c: v["n_questions"] for c, v in res["conditions"].items()}
    assert len(set(n.values())) == 1 and n["icl"] == res["n_test"], n
    assert res["optimizer_steps"] == job.get("epochs", 200)
    assert res["checks"]["question_ids_aligned"] and res["checks"]["C0_unchanged"]


def load_model():
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", cache_dir="./.cache")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("openai-community/gpt2-large", attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
                                        summary_first_dropout=0.0, _attn_implementation="sdpa")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large", config=config, torch_dtype=torch.bfloat16,
                                            cache_dir="./.cache").cuda().eval()
    model.requires_grad_(False)
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
        res["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=sys.version.split()[0],
                          attention="sdpa (production)", shared_gpu_note=job.get("shared_gpu_note"))
        validate(res, job)
        os.makedirs(os.path.dirname(job["out"]) or ".", exist_ok=True)
        tmp = job["out"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, job["out"])
        acc = {c: round(v["accuracy"] * 100, 2) for c, v in res["conditions"].items()}
        print(f"[{n + 1}/{len(todo)}] {job['id']} n_test={res['n_test']} acc={acc} "
              f"adapt={res['timing']['adapt_s']:.1f}s wall={res['wall_s']:.0f}s"
              + (f" checks={res['checks']}" if job.get("smoke") else ""), flush=True)


if __name__ == "__main__":
    main()
