"""Accuracy round 2 runner for NLP-LR (GPT-2 Large): answer-shift calibration scores and the candidate ranking loss.

Guideline: docs/experiment-guideline/CTKV_Accuracy_Plus7_Experiment_Guideline.md. One CT-KV trajectory per job
(study_nlplr.run_job's production-dense training: prefill, leave-one-out + 5% KV token dropout, full-batch AdamW 1e-3,
cosine schedule over `epochs` = 200, grad clip 1.0), with two additions:

  rank_weight > 0   L = L_orig + rank_weight * L_rank, where for every held-out demonstration i
                    L_rank = (1/n) sum_i [logsumexp_{y in Y_i} s_i(y) - s_i(y_i+)],  s = -(mean answer-token CE), FP32.
                    Y_i = the task's label set (classification) or the demonstration's own options (multiple choice),
                    de-duplicated by answer token sequence. All candidates of demonstration i share its row of the KV
                    mask (leave-one-out span + the same token-dropout draw). The gradient is computed exactly in two
                    passes (softmax weights without grad over ALL candidates, then per-chunk backward of
                    sum_y (p_i(y) - 1[y = y+]) s_i(y) / n), so memory does not grow with the number of options.
                    rank_weight == 0 skips the branch entirely (the original trajectory).
  null scores       at every checkpoint the cache also scores "null" questions (content replaced by N/A, same answer
                    candidates), de-duplicated by (rendered null input, candidate text). Calibration itself
                    (s* = s_Ct(x, y) - lambda * [s_Ct(null, y) - s_C0(null, y)]) is applied offline by round2_analyze.py
                    from the stored FP32 scores, so one trajectory serves every lambda.

Every checkpoint stores, per question and option, the legacy loss (the original evaluator: fp16 logits, used for the
original-scoring conditions) and the FP32 loss from the same forward. Test gold labels are only copied to the output
for later statistics; they never enter scoring or training.

Job fields: id, out, task, demos, seed, epochs, rank_weight, checkpoints (steps; 0 = C_0), stop_at, smoke,
timing_only (no test evaluation, no null scoring; per-step times only), null_templates.

Usage (from context-tuning/): .venv/bin/python round2_nlplr.py JOBS.jsonl [--only PREFIX]
"""
import argparse
import gc
import json
import os
import re
import sys
import time
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import get_cosine_schedule_with_warmup

from data_utils import GSDataset, collate_fn_gs, pad_sequence_with_side, parse_pairs, tokenize
from stage3_nlplr import checksum, clone_cache, load_model, rng_state, set_rng_state
from study_nlplr import (EVAL_BS, LR, TOKEN_DROPOUT, build_eval, is_classification, load_tests, option_losses,
                         set_all_seeds)
from train import get_individual_loss

CAND_CHUNK = 16
NULL_TEMPLATES = {"na": "N/A", "dots": "…"}
SENT_PAIR = re.compile(r"^sentence 1: .* \[SEP\] sentence 2: .*$", re.S)
COPA = re.compile(r"^(Cause|Effect): ")


def null_input(task, text, filler):
    """The question with every content field replaced by `filler`; the task format and answer prefix are kept."""
    if task in ("glue-wnli", "sick"):
        assert SENT_PAIR.match(text), (task, text[:80])
        return f"sentence 1: {filler} [SEP] sentence 2: {filler}"
    if task == "superglue-copa":
        m = COPA.match(text)
        assert m, text[:80]
        return f"{m.group(1)}: {filler}"
    assert "[SEP]" not in text, (task, text[:80])
    return filler


def fp32_seq_loss(logits, labels):
    """Mean answer-token CE per sequence in FP32 (same shift and mask as train.get_individual_loss). Keeps the graph."""
    tgt = labels[:, 1:]
    valid = tgt != -100
    sel = logits[:, :-1][valid].float()
    ce = torch.nn.functional.cross_entropy(sel, tgt[valid], reduction="none")
    rows = torch.arange(tgt.shape[0], device=tgt.device).unsqueeze(1).expand_as(tgt)[valid]
    tot = torch.zeros(tgt.shape[0], device=tgt.device, dtype=torch.float32).index_add(0, rows, ce)
    return tot / valid.sum(1).float()


@torch.no_grad()
def score_items(model, kv, prefix_len, items, pad_id):
    """(legacy fp16-logit losses, FP32 losses) of every item under kv; the legacy numbers equal option_losses()."""
    device = kv[0][0].device
    legacy, fp32 = [], []
    for start in range(0, len(items), EVAL_BS):
        batch = items[start:start + EVAL_BS]
        bs = len(batch)
        gen_ids = pad_sequence_with_side([x["gen_input_ids"] for x in batch], pad_id, "right").to(device)
        gen_mask = pad_sequence_with_side([x["gen_attention_mask"] for x in batch], 0, "right").to(device)
        gen_labels = pad_sequence_with_side([x["gen_label_ids"] for x in batch], -100, "right").to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            bkv = [(k.detach().clone().expand(bs, *k.shape[1:]), v.detach().clone().expand(bs, *v.shape[1:])) for k, v in kv]
            kv_mask = torch.ones((bs, bkv[0][0].shape[2]), device=device, dtype=torch.int64)
            embeds = model.transformer.wte(gen_ids)
            mask = torch.cat([kv_mask, gen_mask], dim=1)
            pos = torch.zeros(gen_ids.shape, device=device, dtype=torch.int64)
            for row, n in zip(pos, gen_mask.sum(1)):
                row[:n] = torch.arange(prefix_len, prefix_len + n, device=device)
            out = model(inputs_embeds=embeds, attention_mask=mask, past_key_values=bkv, position_ids=pos)
            legacy.extend(get_individual_loss(lm_logits=out.logits.half(), label_ids=gen_labels).tolist())
        fp32.extend(fp32_seq_loss(out.logits, gen_labels).tolist())
    return legacy, fp32


def build_null(task, tokenizer, demos, tests_by_id, questions, templates):
    """Null items de-duplicated by (rendered null input, candidate text), and the per-question index into them."""
    delim = tokenize(" ", tokenizer)
    items, index, qmap = [], {}, {t: [] for t in templates}
    for tname in templates:
        for q in questions:
            ni = null_input(task, tests_by_id[q["test_id"]]["input"], NULL_TEMPLATES[tname])
            idxs = []
            for opt in q["options"]:
                key = (tname, ni, opt)
                if key not in index:
                    pair = dict(tests_by_id[q["test_id"]], input=ni, output=opt)
                    out = parse_pairs(pairs=demos + [pair], tokenizer=tokenizer, delimiter_token_id=delim, is_train=False)
                    assert out is not None
                    index[key] = len(items)
                    items.append(dict(template=tname, null_input=ni, option=opt, gen_input_ids=out[3],
                                      gen_attention_mask=out[4], gen_label_ids=out[5]))
                idxs.append(index[key])
            qmap[tname].append(idxs)
    return items, qmap


def build_candidates(task, demos, gs, tokenizer):
    """Per demonstration (by example_idx): the non-gold candidates in training format, de-duplicated by token ids."""
    cls = is_classification(task)
    label_space = json.load(open(f"config/tasks/{task}.json")).get("options") if cls else None
    cands, info = {}, {}
    for ex in gs.parsed_examples:
        i = ex["example_idx"]
        d = demos[i]
        opts = list(label_space) if cls else list(d["options"])
        gold_tok = tuple(tokenize(d["output"], tokenizer).tolist())
        seen, rows, note = {gold_tok}, [], {}
        toks = [tuple(tokenize(o, tokenizer).tolist()) for o in opts]
        if gold_tok not in toks:
            note["gold_not_in_options"] = True
        if len(set(toks)) < len(toks):
            note["duplicate_candidates_merged"] = len(toks) - len(set(toks))
        for o, tk in zip(opts, toks):
            if tk in seen:
                continue
            seen.add(tk)
            f = gs.format(i, dict(d, output=o))
            if f is None:
                note["candidate_overflow"] = note.get("candidate_overflow", 0) + 1
                continue
            rows.append(dict(option=o, input_ids=f["input_ids"], attention_mask=f["attention_mask"],
                             label_ids=f["label_ids"]))
        if not rows:
            note["single_valid_candidate"] = True
        cands[i] = rows
        info[i] = dict(n_candidates=len(rows) + 1, **note)
    return cands, info


def forward_train(model, kv, P, ids, amask, labels, kv_mask, with_labels):
    bs = ids.shape[0]
    bkv = tuple((k.expand(bs, -1, -1, -1), v.expand(bs, -1, -1, -1)) for k, v in kv)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pos = torch.zeros((bs, ids.shape[1]), device=ids.device, dtype=torch.int64)
        for row, n in zip(pos, amask.sum(1)):
            row[:n] = torch.arange(P, P + n, device=ids.device)
        embeds = model.transformer.wte(ids)
        full_mask = torch.cat([kv_mask, amask], dim=1)
        return model(inputs_embeds=embeds, attention_mask=full_mask, labels=labels if with_labels else None,
                     use_cache=True, past_key_values=bkv, position_ids=pos)


def cand_chunks(flat, pad_id, device):
    for s in range(0, len(flat), CAND_CHUNK):
        ch = flat[s:s + CAND_CHUNK]
        ids = pad_sequence_with_side([c["input_ids"] for c in ch], pad_id, "right").to(device)
        am = pad_sequence_with_side([c["attention_mask"] for c in ch], 0, "right").to(device)
        lb = pad_sequence_with_side([c["label_ids"] for c in ch], -100, "right").to(device)
        yield s, ch, ids, am, lb


def rank_scores_nograd(model, kv, P, flat, rows_of, kv_mask, pad_id, check_span=None):
    """FP32 scores (-loss) of the flattened non-gold candidates; each uses the KV-mask row of its demonstration."""
    out = torch.empty(len(flat), dtype=torch.float32, device=kv_mask.device)
    with torch.no_grad():
        for s, ch, ids, am, lb in cand_chunks(flat, pad_id, kv_mask.device):
            m = kv_mask[rows_of[s:s + len(ch)]]
            if check_span is not None:
                for r, c in zip(m, ch):
                    a, b = check_span[c["demo"]]
                    assert int(r[a:b].sum()) == 0
            o = forward_train(model, kv, P, ids, am, lb, m, False)
            out[s:s + len(ch)] = -fp32_seq_loss(o.logits, lb)
            del o
    return out


def rank_terms(gold_s, cand_s, owner, n_rows):
    """Per batch row: logsumexp over its candidates (gold + non-gold), softmax weights, margin, gold rank."""
    lse, p_gold, p_cand = torch.empty(n_rows), torch.empty(n_rows), torch.zeros(len(cand_s))
    margin, rank = torch.full((n_rows,), float("nan")), torch.ones(n_rows)
    g, c = gold_s.detach().float().cpu(), cand_s.detach().float().cpu()
    for b in range(n_rows):
        sel = (owner == b).nonzero().flatten()
        s = torch.cat([g[b:b + 1], c[sel]])
        lse[b] = torch.logsumexp(s, 0)
        p = torch.softmax(s, 0)
        p_gold[b] = p[0]
        p_cand[sel] = p[1:]
        if len(sel):
            margin[b] = g[b] - c[sel].max()
            rank[b] = 1 + int((c[sel] > g[b]).sum())
    return lse, p_gold, p_cand, margin, rank


def run_job(job, model, tokenizer):
    set_all_seeds(job.get("seed", 0))
    task, demos = job["task"], job["demos"]
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    T = job.get("epochs", 200)
    w_rank = float(job.get("rank_weight", 0.0))
    timing_only = bool(job.get("timing_only"))
    smoke = bool(job.get("smoke"))
    cps = sorted(set(job.get("checkpoints", [0, 50, 200])))
    stop_at = job.get("stop_at") or max(cps)
    templates = job.get("null_templates", ["na", "dots"])
    assert job.get("loss", "raw") == "raw" and stop_at <= T and max(cps) <= stop_at
    res = dict(id=job["id"], task=task, loss="raw", rank_weight=w_rank, epochs=T, stop_at=stop_at, checkpoint_steps=cps,
               meta=job.get("meta", {}), stage1_id=job.get("stage1_id"), demo_pool_ids=[d.get("pool_id") for d in demos],
               demo_labels=[d["output"] for d in demos], checkpoints={}, timing={}, checks={},
               scheduler=dict(type="cosine_with_warmup", warmup=0, num_training_steps=T, lr=LR, optimizer="AdamW",
                              weight_decay=0.0, clip=1.0, token_dropout=TOKEN_DROPOUT, batch="full"))

    t0 = time.perf_counter()
    tests = [] if timing_only else load_tests(task)
    prefix_ids, starts, items = build_eval(tokenizer, demos, tests)
    P = int(prefix_ids.shape[0])
    tests_by_id = dict(tests)
    questions, slices = [], []
    for n, it in enumerate(items):
        if not questions or questions[-1]["test_id"] != it["test_id"]:
            questions.append(dict(test_id=it["test_id"], gold=it["correct"], options=[]))
            slices.append([n, n])
        questions[-1]["options"].append(it["option"])
        slices[-1][1] = n + 1
    assert len({q["test_id"] for q in questions}) == len(questions)
    null_items, qmap = build_null(task, tokenizer, demos, tests_by_id, questions, templates)
    res["questions"] = questions
    res["null_items"] = [[x["template"], x["null_input"], x["option"]] for x in null_items]
    res["question_null_idx"] = qmap
    res["n_test"], res["prefix_len"] = len(questions), P
    res["timing"]["build_s"] = time.perf_counter() - t0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_adapt = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kv = model(input_ids=prefix_ids.unsqueeze(0).to(device)).past_key_values
    kv = tuple((k.detach().to(torch.bfloat16).clone(), v.detach().to(torch.bfloat16).clone()) for k, v in kv)
    torch.cuda.synchronize()
    res["timing"]["prefill_s"] = time.perf_counter() - t_adapt
    c0 = clone_cache(kv) if smoke else None
    c0_sum = checksum(kv) if smoke else None
    eval_time = 0.0
    kv_len = kv[0][0].shape[2]

    def adapt_so_far():
        torch.cuda.synchronize()
        return time.perf_counter() - t_adapt - eval_time

    def checkpoint(step):
        nonlocal eval_time
        cum = adapt_so_far()
        s = time.perf_counter()
        before = checksum(kv) if smoke else None
        cp = dict(step=step, cumulative_adapt_s=cum)
        if not timing_only:
            if job.get("eval_real", True) or step > 0:
                a = time.perf_counter()
                leg, f32 = score_items(model, kv, P, items, pad_id)
                torch.cuda.synchronize()
                cp["real_eval_s"] = time.perf_counter() - a
                cp["legacy_loss"] = [leg[i:j] for i, j in slices]
                cp["fp32_loss"] = [f32[i:j] for i, j in slices]
                pred = [q["options"][int(np.argmin(l))] for q, l in zip(questions, cp["legacy_loss"])]
                cp["accuracy"] = float(np.mean([p == q["gold"] for p, q in zip(pred, questions)])) if pred else None
            a = time.perf_counter()
            leg, f32 = score_items(model, kv, P, null_items, pad_id)
            torch.cuda.synchronize()
            cp["null_eval_s"] = time.perf_counter() - a
            cp["null_legacy_loss"], cp["null_fp32_loss"] = leg, f32
            cp["n_null_items"] = len(null_items)
        if smoke:
            res["checks"][f"cache_unchanged_by_scoring@{step}"] = checksum(kv) == before
        res["checkpoints"][str(step)] = cp
        torch.cuda.synchronize()
        eval_time += time.perf_counter() - s

    if 0 in cps:
        checkpoint(0)
    params = [x for layer in kv for x in layer]
    for p in params:
        p.requires_grad_(True)
    gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(demos)}, tokenizer=tokenizer, pad_side="right",
                   past_kv_len=P)
    batch_size = len(gs)
    loader = DataLoader(gs, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_gs, dataset=gs),
                        drop_last=False, num_workers=0)
    cands, cand_info = build_candidates(task, demos, gs, tokenizer) if (w_rank > 0 or smoke) else ({}, {})
    res["rank_candidates"] = {str(k): v for k, v in cand_info.items()}
    span = {i: (starts[i], starts[i + 1] if i < len(starts) - 1 else kv_len) for i in range(len(demos))}
    optim = torch.optim.AdamW([{"params": params, "lr": LR}], weight_decay=0.0)
    sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=T)
    model.train()
    n_updates = 0
    losses, rank_losses, rank_margin, rank_acc, step_s, rank_s = [], [], [], [], [], []

    def flatten(ex_idx):
        flat, rows_of = [], []
        for b, idx in enumerate(ex_idx):
            for c in cands[idx]:
                flat.append(dict(c, demo=idx))
                rows_of.append(b)
        return flat, torch.tensor(rows_of, dtype=torch.long, device=device)

    def rank_backward(gold_s, flat, rows_of, kv_mask, weight, do_check):
        """Adds weight * dL_rank/dKV to .grad (the gold term stays attached to gold_s and is returned)."""
        owner = rows_of.cpu()
        cs = rank_scores_nograd(model, kv, P, flat, rows_of, kv_mask, pad_id, span if do_check else None)
        lse, p_gold, p_cand, margin, rank = rank_terms(gold_s, cs, owner, len(gold_s))
        valid = torch.tensor([bool((owner == b).any()) for b in range(len(gold_s))])
        n = max(int(valid.sum()), 1)
        l_rank = float(((lse - gold_s.detach().float().cpu())[valid]).sum() / n)
        for s, ch, ids, am, lb in cand_chunks(flat, pad_id, device):
            o = forward_train(model, kv, P, ids, am, lb, kv_mask[rows_of[s:s + len(ch)]], False)
            sc = -fp32_seq_loss(o.logits, lb)
            (weight * (p_cand[s:s + len(ch)].to(device) * sc).sum() / n).backward()
            del o, sc
        gold_term = weight * (((p_gold - 1.0) * valid.float()).to(device) * gold_s).sum() / n
        return gold_term, l_rank, margin[valid], rank[valid], cs

    for step in range(1, stop_at + 1):
        torch.cuda.synchronize()
        t_step = time.perf_counter()
        step_loss, t_rank = 0.0, 0.0
        for batch in loader:
            ids = batch["input_ids"].to(device)
            amask = batch["attention_mask"].to(device)
            labels = batch["label_ids"].to(device)
            ex_idx = [int(i) for i in batch["example_idx"]]
            bs = ids.shape[0]
            assert bs == batch_size
            kv_mask = torch.ones((bs, kv_len), device=device, dtype=torch.int64)
            for b, idx in enumerate(ex_idx):
                kv_mask[b, span[idx][0]:span[idx][1]] = 0
            if TOKEN_DROPOUT:
                drop = (torch.rand_like(kv_mask, dtype=torch.float) > TOKEN_DROPOUT).float()
                kv_mask = (kv_mask * drop).long()
            if smoke and step == 1:
                st = rng_state()
                flat, rows_of = flatten(ex_idx)
                # rank-only gradient (weight 1, no original loss), scorer agreement, permutation invariance
                o = forward_train(model, kv, P, ids, amask, labels, kv_mask, False)
                gs_ = -fp32_seq_loss(o.logits, labels)
                gterm, l1, _, _, cs1 = rank_backward(gs_, flat, rows_of, kv_mask, 1.0, True)
                gterm.backward()
                del o
                gn = [float(p.grad.float().norm()) for p in params]
                res["checks"]["rank_only_L_finite"] = bool(np.isfinite(l1))
                res["checks"]["rank_only_grad_nonzero_tensors"] = int(sum(g > 0 for g in gn))
                res["checks"]["rank_only_grad_norm_total"] = float(np.sqrt(sum(g * g for g in gn)))
                res["checks"]["rank_only_L"] = l1
                optim.zero_grad()
                perm = torch.randperm(len(flat), generator=torch.Generator().manual_seed(1)).tolist()
                flat2, rows2 = [flat[i] for i in perm], rows_of[perm]
                cs2 = rank_scores_nograd(model, kv, P, flat2, rows2, kv_mask, pad_id)
                lse2 = rank_terms(gs_, cs2, rows2.cpu(), bs)[0]
                lse1 = rank_terms(gs_, cs1, rows_of.cpu(), bs)[0]
                res["checks"]["permuted_candidates_max_abs_dlogsumexp"] = float((lse1 - lse2).abs().max())
                inv = torch.empty(len(perm), dtype=torch.long)
                inv[torch.tensor(perm)] = torch.arange(len(perm))
                res["checks"]["permuted_candidates_max_abs_dscore"] = float((cs1.cpu() - cs2.cpu()[inv]).abs().max()) \
                    if len(perm) else 0.0
                res["checks"]["model_weights_without_grad"] = all(p.grad is None and not p.requires_grad
                                                                  for p in model.parameters())
                # tensor scorer vs the legacy evaluator: eval mode, same cache, same inputs (all demos' candidates)
                model.eval()
                ev = [dict(gen_input_ids=c["input_ids"], gen_attention_mask=c["attention_mask"],
                           gen_label_ids=c["label_ids"]) for c in flat]
                ones = torch.ones((len(flat), kv_len), device=device, dtype=torch.int64) if flat else kv_mask[:0]
                old = option_losses(model, kv, P, ev, pad_id)
                leg, f32 = score_items(model, kv, P, ev, pad_id)
                new = (-rank_scores_nograd(model, kv, P, flat, torch.arange(len(flat), device=device), ones, pad_id)).tolist()
                res["checks"]["scorer_legacy_equals_option_losses_max_abs"] = max([abs(a - b) for a, b in zip(old, leg)] or [0])
                res["checks"]["scorer_tensor_vs_legacy_max_abs"] = max([abs(a - b) for a, b in zip(old, new)] or [0])
                res["checks"]["scorer_tensor_vs_fp32_eval_max_abs"] = max([abs(a - b) for a, b in zip(f32, new)] or [0])
                agree, near = 0, 0
                for b in range(bs):
                    sel = [k for k in range(len(flat)) if int(rows_of[k]) == b]
                    if len(sel) > 1:
                        agree += int(np.argsort([old[k] for k in sel]).tolist() == np.argsort([new[k] for k in sel]).tolist())
                        so = sorted(old[k] for k in sel)
                        near += int(min(b2 - a2 for a2, b2 in zip(so, so[1:])) < 1e-3)
                res["checks"]["scorer_rank_agreement"] = [agree, sum(1 for b in range(bs) if (rows_of == b).sum() > 1), near]
                model.train()
                set_rng_state(st)
            out = forward_train(model, kv, P, ids, amask, labels, kv_mask, True)
            loss = out.loss * bs / batch_size
            if w_rank > 0:
                torch.cuda.synchronize()
                a = time.perf_counter()
                flat, rows_of = flatten(ex_idx)
                gold_s = -fp32_seq_loss(out.logits, labels)
                gterm, l_rank, margin, rank, _ = rank_backward(gold_s, flat, rows_of, kv_mask, w_rank, smoke and step == 1)
                (loss + gterm).backward()
                torch.cuda.synchronize()
                t_rank += time.perf_counter() - a
                rank_losses.append(l_rank)
                rank_margin.append(float(margin.mean()) if len(margin) else float("nan"))
                rank_acc.append(float((rank == 1).float().mean()) if len(rank) else float("nan"))
            else:
                loss.backward()
            step_loss += loss.item()
            del out
        losses.append(step_loss)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optim.step()
        sched.step()
        optim.zero_grad()
        n_updates += 1
        torch.cuda.synchronize()
        step_s.append(time.perf_counter() - t_step)
        rank_s.append(t_rank)
        if step in cps:
            st, was_training = rng_state(), model.training
            model.eval()
            for p in params:
                p.requires_grad_(False)
            checkpoint(step)
            for p in params:
                p.requires_grad_(True)
            model.train(was_training)
            if smoke:
                now = rng_state()
                res["checks"][f"rng_unchanged_by_scoring@{step}"] = bool(
                    torch.equal(st[2], now[2]) and all(torch.equal(a, b) for a, b in zip(st[3], now[3])))
                res["checks"][f"train_mode_restored@{step}"] = model.training
            set_rng_state(st)
    model.eval()
    for p in params:
        p.requires_grad_(False)
    res["optimizer_steps"] = n_updates
    res["scheduler"]["last_lr"] = sched.get_last_lr()[0]
    res["scheduler"]["scheduler_steps"] = sched.last_epoch
    del optim, sched
    if smoke:
        res["checks"]["C0_copy_unchanged"] = checksum(c0) == c0_sum
    res["timing"]["adapt_s"] = adapt_so_far()
    res["timing"]["eval_s_total"] = eval_time
    res["timing"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["timing"]["step_s_median"] = float(np.median(step_s)) if step_s else None
    res["timing"]["step_s_median_after_warmup"] = float(np.median(step_s[3:])) if len(step_s) > 3 else None
    res["timing"]["rank_s_median"] = float(np.median(rank_s)) if step_s else None
    res["timing"]["step_s_sum"] = float(np.sum(step_s))
    res["train_losses"], res["rank_losses"] = losses, rank_losses
    res["rank_train_margin"], res["rank_train_acc"] = rank_margin, rank_acc
    res["n_rank_candidates_total"] = int(sum(len(v) for v in cands.values()))
    del kv, c0
    gc.collect()
    torch.cuda.empty_cache()
    return res


def validate(res, job):
    assert res["optimizer_steps"] == res["stop_at"] == res["scheduler"]["scheduler_steps"]
    assert sorted(int(k) for k in res["checkpoints"]) == res["checkpoint_steps"]
    if job.get("timing_only"):
        return
    nq = res["n_test"]
    for k, cp in res["checkpoints"].items():
        assert len(cp["null_fp32_loss"]) == len(res["null_items"]) and np.isfinite(cp["null_fp32_loss"]).all(), k
        if "fp32_loss" in cp:
            assert len(cp["fp32_loss"]) == nq and all(len(a) == len(q["options"]) for a, q in zip(cp["fp32_loss"], res["questions"]))
            assert all(np.isfinite(a).all() for a in cp["fp32_loss"]), k
    assert all(len(v) == nq for v in res["question_null_idx"].values())
    if res["rank_weight"] > 0:
        assert len(res["rank_losses"]) == res["stop_at"] and np.isfinite(res["rank_losses"]).all()


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
        for attempt in range(3):
            try:
                res = run_job(job, model, tokenizer)
                break
            except torch.cuda.OutOfMemoryError:
                if attempt == 2:
                    raise
                print(f"OOM on {job['id']}, retry {attempt + 1} after 60 s", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
                time.sleep(60)
        res["wall_s"] = time.time() - t
        res["env"] = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(), python=sys.version.split()[0],
                          attention="sdpa (production)", cand_chunk=CAND_CHUNK)
        validate(res, job)
        os.makedirs(os.path.dirname(job["out"]) or ".", exist_ok=True)
        tmp = job["out"] + ".tmp"
        with open(tmp, "w") as f:
            json.dump(res, f)
        os.replace(tmp, job["out"])
        acc = {k: round(v["accuracy"] * 100, 2) for k, v in res["checkpoints"].items() if v.get("accuracy") is not None}
        print(f"[{n + 1}/{len(todo)}] {job['id']} n_test={res['n_test']} acc={acc} adapt={res['timing']['adapt_s']:.1f}s "
              f"step={res['timing']['step_s_median']} wall={res['wall_s']:.0f}s"
              + (f" checks={res['checks']}" if job.get("smoke") else ""), flush=True)


if __name__ == "__main__":
    main()
