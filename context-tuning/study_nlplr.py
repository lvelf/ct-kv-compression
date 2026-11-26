"""CT-KV accuracy study on NLP-LR (GPT-2 Large): run a list of single-task CT-KV trajectories.

Each job is one task, one explicit demonstration list, one loss, and a set of checkpoints evaluated on the
same 16-shot-style test set inside one training trajectory (cosine schedule over `epochs` steps; evaluating a
checkpoint does not touch the training RNG). Jobs are read from a JSONL file; a job whose output JSON already
exists is skipped, so the runner can be restarted at any time.

Job fields:
  id, out                  unique id and output JSON path
  task                     MetaICL task name
  demos                    list of {"input", "output", "options", "pool_id"} in prompt order
  loss                     "raw" (the paper's token-mean over all answer tokens) or "balanced"
                           ((1/C) * sum_c mean_{i in class c} of the per-example token-mean CE)
  epochs                   length of the cosine schedule (T_max); 200 in the paper
  checkpoints              steps at which to evaluate on the test set (0 = ICL with the same cache)
  stop_at                  optional: stop training after this many steps (keeps the T_max=epochs schedule)
  dev                      optional list of {"input", "output", "options", "dev_id"}; mean answer-token NLL is
                           recorded at dev_checkpoints (forward only, never used for gradients)
  dev_checkpoints          steps at which to score dev
  eval_test                default true; false skips test evaluation (selection-only runs)
  seed                     torch/numpy seed for the trajectory (data order and token dropout)
  meta                     free-form; copied to the output

Usage (from context-tuning/): .venv/bin/python study_nlplr.py JOBS.jsonl [--only PREFIX]
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
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel, get_cosine_schedule_with_warmup

from data_utils import GSDataset, collate_fn_gs, parse_pairs, tokenize, pad_sequence_with_side
from train import compute_macrof1_or_accuracy, get_individual_loss

DATA_DIR = "./metaicl-data/data"
TEST_SPLIT = "87"  # the test files are identical across splits 13/21/42/87/100
EVAL_BS = 16
LR = 1e-3
TOKEN_DROPOUT = 0.05


def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_classification(task):
    return json.load(open(f"config/tasks/{task}.json"))["task_type"] == "classification"


def load_tests(task):
    """Test examples in the order the original EvalDataset visits them (RandomState(0) shuffle)."""
    lines = open(os.path.join(DATA_DIR, task, f"{task}_16_{TEST_SPLIT}_test.jsonl")).readlines()
    order = list(range(len(lines)))
    np.random.RandomState(0).shuffle(order)
    return [(i, json.loads(lines[i])) for i in order]


def build_eval(tokenizer, demos, tests):
    """Return (demon_input_ids, demon_start_idxs, items) with the original length filter and patience rule.

    items: list of dicts with test_id, option, correct, gen tensors."""
    delim = tokenize(" ", tokenizer)
    items, demon_input_ids, demon_start_idxs = [], None, None
    patience = 0
    for test_id, test in tests:
        outs = []
        for option in test["options"]:
            pair = dict(test, output=option)
            out = parse_pairs(pairs=demos + [pair], tokenizer=tokenizer, delimiter_token_id=delim, is_train=False)
            outs.append(None if out is None else (option, out))
        if any(o is None for o in outs):
            patience += 1
            if patience == 100:
                break
            continue
        patience = 0
        for option, out in outs:
            _, _, _, gen_ids, gen_mask, gen_labels, d_ids, _, d_starts = out
            demon_input_ids, demon_start_idxs = d_ids, d_starts
            items.append(dict(test_id=test_id, option=option, correct=test["output"], gen_input_ids=gen_ids,
                              gen_attention_mask=gen_mask, gen_label_ids=gen_labels))
    if demon_input_ids is None:  # no test evaluation (or every test too long): build the prefix alone
        tiny = {"input": "a", "output": "b", "options": ["b"], "task": demos[0].get("task")}
        out = parse_pairs(pairs=demos + [tiny], tokenizer=tokenizer, delimiter_token_id=delim, is_train=False)
        if out is None:
            raise ValueError("demonstrations alone exceed the 1024-token context")
        demon_input_ids, demon_start_idxs = out[6], out[8]
    return demon_input_ids, demon_start_idxs, items


@torch.no_grad()
def option_losses(model, kv, prefix_len, items, pad_id):
    """Mean answer-token loss of every item under the cache kv (the original evaluation forward)."""
    device = kv[0][0].device
    losses = []
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
            losses.extend(get_individual_loss(lm_logits=out.logits.half(), label_ids=gen_labels).tolist())
    return losses


def evaluate(model, kv, prefix_len, items, pad_id, task):
    losses = option_losses(model, kv, prefix_len, items, pad_id)
    per_test = {}
    for item, loss in zip(items, losses):
        per_test.setdefault(item["test_id"], []).append((item["option"], loss, item["correct"]))
    preds, golds, rows = [], [], []
    for test_id, opts in per_test.items():
        pred = min(opts, key=lambda x: x[1])[0]
        preds.append(pred)
        golds.append(opts[0][2])
        rows.append([test_id, pred, opts[0][2], [round(l, 4) for _, l, _ in opts]])
    score = compute_macrof1_or_accuracy(preds, golds, is_classification(task)) if rows else float("nan")
    return score, rows


def dev_nll(model, kv, prefix_len, tokenizer, demos, dev, pad_id):
    """Mean over dev questions of the token-mean NLL of the gold answer (forward only)."""
    delim = tokenize(" ", tokenizer)
    items = []
    for d in dev:
        out = parse_pairs(pairs=demos + [d], tokenizer=tokenizer, delimiter_token_id=delim, is_train=False)
        if out is None:
            continue
        items.append(dict(gen_input_ids=out[3], gen_attention_mask=out[4], gen_label_ids=out[5]))
    losses = option_losses(model, kv, prefix_len, items, pad_id)
    return float(np.mean(losses)) if losses else float("nan"), len(items)


def balanced_weights(demos, parsed_idx):
    """Per-example weight w_i so that sum_i w_i * tokenmean_i = (1/C) sum_c mean_{i in c} tokenmean_i."""
    labels = [demos[i]["output"] for i in parsed_idx]
    classes = sorted(set(labels))
    counts = {c: labels.count(c) for c in classes}
    return {i: 1.0 / (len(classes) * counts[demos[i]["output"]]) for i in parsed_idx}, counts


def run_job(job, model, tokenizer):
    set_all_seeds(job.get("seed", 0))
    task, demos = job["task"], job["demos"]
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    epochs = job.get("epochs", 200)
    checkpoints = sorted(set(job.get("checkpoints", [0, epochs])))
    dev_cps = sorted(set(job.get("dev_checkpoints", [])))
    stop_at = job.get("stop_at") or epochs
    eval_test = job.get("eval_test", True)
    res = dict(id=job["id"], task=task, loss=job["loss"], epochs=epochs, stop_at=stop_at, meta=job.get("meta", {}),
               demo_pool_ids=[d.get("pool_id") for d in demos], demo_labels=[d["output"] for d in demos],
               checkpoints={}, dev={}, timing={})

    t0 = time.perf_counter()
    tests = load_tests(task) if eval_test else []
    prefix_ids, starts, items = build_eval(tokenizer, demos, tests)
    res["n_test"] = len({x["test_id"] for x in items})
    res["prefix_len"] = int(prefix_ids.shape[0])
    res["timing"]["build_s"] = time.perf_counter() - t0

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t_adapt = time.perf_counter()
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kv = model(input_ids=prefix_ids.unsqueeze(0).to(device)).past_key_values
    kv = tuple((k.detach().to(torch.bfloat16).clone(), v.detach().to(torch.bfloat16).clone()) for k, v in kv)
    eval_time = 0.0

    def checkpoint(step):
        nonlocal eval_time
        torch.cuda.synchronize()
        s = time.perf_counter()
        if eval_test and step in checkpoints:
            score, rows = evaluate(model, kv, prefix_ids.shape[0], items, pad_id, task)
            res["checkpoints"][str(step)] = dict(score=score, predictions=rows)
        if step in dev_cps:
            nll, n = dev_nll(model, kv, prefix_ids.shape[0], tokenizer, demos, job["dev"], pad_id)
            res["dev"][str(step)] = dict(nll=nll, n=n)
        torch.cuda.synchronize()
        eval_time += time.perf_counter() - s

    checkpoint(0)
    losses = []
    if stop_at > 0 and epochs > 0:
        params = [x for layer in kv for x in layer]
        for p in params:
            p.requires_grad_(True)
        gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(demos)}, tokenizer=tokenizer, pad_side="right",
                       past_kv_len=prefix_ids.shape[0])
        batch_size = len(gs)  # full batch, as with the paper's ct_batch_size=16 and 16 demos
        loader = DataLoader(gs, batch_size=batch_size, shuffle=True, collate_fn=partial(collate_fn_gs, dataset=gs),
                            drop_last=False, num_workers=0)
        parsed_idx = [e["example_idx"] for e in gs.parsed_examples]
        if job["loss"] == "balanced":
            weights, counts = balanced_weights(demos, parsed_idx)
            res["class_counts"] = counts
            if len(counts) < 2:
                res["balanced_fallback_to_raw"] = True
        optim = torch.optim.AdamW([{"params": params, "lr": LR}], weight_decay=0.0)
        sched = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)
        model.train()
        kv_len = kv[0][0].shape[2]
        for step in range(1, stop_at + 1):
            step_loss = 0.0
            for batch in loader:
                ids = batch["input_ids"].to(device)
                amask = batch["attention_mask"].to(device)
                labels = batch["label_ids"].to(device)
                ex_idx = batch["example_idx"]
                bs = ids.shape[0]
                kv_mask = torch.ones((bs, kv_len), device=device, dtype=torch.int64)
                bkv = tuple((k.expand(bs, -1, -1, -1), v.expand(bs, -1, -1, -1)) for k, v in kv)
                for b, idx in enumerate(ex_idx):
                    s = starts[idx]
                    e = starts[idx + 1] if idx < len(starts) - 1 else kv_len
                    kv_mask[b, s:e] = 0
                if TOKEN_DROPOUT:
                    drop = (torch.rand_like(kv_mask, dtype=torch.float) > TOKEN_DROPOUT).float()
                    kv_mask = (kv_mask * drop).long()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    pos = torch.zeros((bs, ids.shape[1]), device=device, dtype=torch.int64)
                    for row, n in zip(pos, amask.sum(1)):
                        row[:n] = torch.arange(prefix_ids.shape[0], prefix_ids.shape[0] + n, device=device)
                    embeds = model.transformer.wte(ids)
                    full_mask = torch.cat([kv_mask, amask], dim=1)
                    if job["loss"] == "raw" or res.get("balanced_fallback_to_raw"):
                        out = model(inputs_embeds=embeds, attention_mask=full_mask, labels=labels, use_cache=True,
                                    past_key_values=bkv, position_ids=pos)
                        loss = out.loss * bs / batch_size
                    else:
                        out = model(inputs_embeds=embeds, attention_mask=full_mask, use_cache=True,
                                    past_key_values=bkv, position_ids=pos)
                        logits = out.logits[:, :-1].float()
                        tgt = labels[:, 1:]
                        ce = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                                         ignore_index=-100, reduction="none").view(tgt.shape)
                        valid = (tgt != -100).float()
                        per_ex = (ce * valid).sum(1) / valid.sum(1)
                        w = torch.tensor([weights[int(i)] for i in ex_idx], device=device, dtype=per_ex.dtype)
                        loss = (per_ex * w).sum()
                loss.backward()
                step_loss += loss.item()
            losses.append(step_loss)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            sched.step()
            optim.zero_grad()
            if step in checkpoints or step in dev_cps:
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
    torch.cuda.synchronize()
    res["timing"]["adapt_s"] = time.perf_counter() - t_adapt - eval_time
    res["timing"]["checkpoint_eval_s"] = eval_time
    res["timing"]["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
    res["train_losses"] = losses
    del kv
    gc.collect()
    torch.cuda.empty_cache()
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("jobs")
    ap.add_argument("--only", default=None, help="run only jobs whose id starts with this prefix")
    a = ap.parse_args()
    jobs = [json.loads(l) for l in open(a.jobs) if l.strip()]
    if a.only:
        jobs = [j for j in jobs if j["id"].startswith(a.only)]
    todo = [j for j in jobs if not os.path.exists(j["out"])]
    print(f"{len(jobs)} jobs, {len(todo)} to run", flush=True)
    if not todo:
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2-large", cache_dir="./.cache")
    tokenizer.pad_token = tokenizer.eos_token
    config = GPT2Config.from_pretrained("openai-community/gpt2-large", attn_pdrop=0.0, embd_pdrop=0.0, resid_pdrop=0.0,
                                        summary_first_dropout=0.0, _attn_implementation="sdpa")
    model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2-large", config=config, torch_dtype=torch.bfloat16,
                                            cache_dir="./.cache").cuda().eval()
    model.requires_grad_(False)
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
        print(f"[{n + 1}/{len(todo)}] {job['id']} n_test={res['n_test']} scores={scores} dev={ {k: round(v['nll'], 4) for k, v in res['dev'].items()} } "
              f"{res['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
