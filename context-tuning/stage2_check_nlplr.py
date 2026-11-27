"""Stage 2 r = 1 equivalence and mask checks for NLP-LR (GPT-2 Large).

For one job (task, demos): on the initial cache, compare option losses and predictions under
  prod   (production SDPA kernel)   vs   dense (diagnostic path, keep every legal position)   vs   topk r = 1
and one training microbatch's KV gradient (leave-one-out + token dropout mask, same dropout draw for all three).
Then with topk r = 0.75 / 0.5 (and fixed / random) verify that no probability lands on a non-selected demo position
or on a leave-one-out / dropped position, that kept counts follow k = min(n, max(1, floor(r n))), and that causal
order is preserved (a query never sees later current-input positions).
Usage (from context-tuning/): .venv/bin/python stage2_check_nlplr.py JOBS.jsonl JOB_ID [JOB_ID ...] --out OUT.json
"""
import argparse
import json
from functools import partial

import torch
from torch.utils.data import DataLoader

import ctkv_sparse as SP
from data_utils import GSDataset, collate_fn_gs
from stage2_nlplr import answer_attention, forward_from_cache, load_model, train_batch_tensors
from study_nlplr import TOKEN_DROPOUT, build_eval, load_tests, option_losses, set_all_seeds


def check_job(job, model, tok):
    set_all_seeds(0)
    device = "cuda"
    prefix_ids, starts, items = build_eval(tok, job["demos"], load_tests(job["task"]))
    P = int(prefix_ids.shape[0])
    owner = torch.full((P,), -1, dtype=torch.long)
    for i, s in enumerate(starts):
        owner[s:(starts[i + 1] if i < len(starts) - 1 else P)] = i
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        kv = model(input_ids=prefix_ids.unsqueeze(0).to(device)).past_key_values
    kv = tuple((k.detach().to(torch.bfloat16).clone(), v.detach().to(torch.bfloat16).clone()) for k, v in kv)
    gs = GSDataset(demonstration_pairs={i: p for i, p in enumerate(job["demos"])}, tokenizer=tok, pad_side="right",
                   past_kv_len=P)
    fixed = answer_attention(model, kv, P, gs, starts, device)
    SP.configure(P, owner >= 0, owner=owner, n_demos=len(starts), fixed_score=fixed, mask_seed=0)
    items = items[:400]
    out = dict(id=job["id"], task=job["task"], P=P, n_items=len(items))
    losses = {}
    for mode, r in (("prod", 1), ("dense", 1), ("topk", 1.0)):
        with SP.sparse(mode, r, check=True):
            SP.STATE.reset_stats()
            losses[f"{mode}"] = torch.tensor(option_losses(model, kv, P, items, tok.pad_token_id))
            if mode != "prod":
                out[f"viol_{mode}"] = SP.STATE.stats()["violations"]

    def preds(l):
        per = {}
        for it, x in zip(items, l.tolist()):
            per.setdefault(it["test_id"], []).append((x, it["option"]))
        return {t: min(v)[1] for t, v in per.items()}

    p0 = preds(losses["prod"])
    for m in ("dense", "topk"):
        d = (losses[m] - losses["prod"]).abs()
        pm = preds(losses[m])
        out[f"loss_maxabs_{m}_vs_prod"] = float(d.max())
        out[f"loss_meanabs_{m}_vs_prod"] = float(d.mean())
        out[f"pred_flips_{m}_vs_prod"] = sum(pm[t] != p0[t] for t in p0)
    out["loss_maxabs_topk1_vs_dense"] = float((losses["topk"] - losses["dense"]).abs().max())
    out["n_tests"] = len(p0)

    # one-microbatch KV gradient with the training mask (same dropout draw for all modes)
    loader = DataLoader(gs, batch_size=len(gs), shuffle=False, collate_fn=partial(collate_fn_gs, dataset=gs))
    batch = next(iter(loader))
    torch.manual_seed(123)
    ids, amask, labels, kv_mask = train_batch_tensors(batch, starts, P, device, TOKEN_DROPOUT)
    grads = {}
    for mode, r in (("prod", 1), ("dense", 1), ("topk", 1.0)):
        params = [x.detach().clone().requires_grad_(True) for layer in kv for x in layer]
        kvp = tuple((params[2 * i], params[2 * i + 1]) for i in range(len(kv)))
        with torch.autocast("cuda", dtype=torch.bfloat16), SP.sparse(mode, r):
            o = forward_from_cache(model, kvp, P, ids, amask, kv_mask, labels=labels)
        o.loss.backward()
        grads[mode] = torch.cat([p.grad.float().flatten() for p in params])
    for m in ("dense", "topk"):
        g, g0 = grads[m], grads["prod"]
        out[f"grad_cos_{m}_vs_prod"] = float(torch.nn.functional.cosine_similarity(g, g0, dim=0))
        out[f"grad_relerr_{m}_vs_prod"] = float((g - g0).norm() / g0.norm())

    # masks under sparse budgets: training mask (leave-one-out + dropout) and eval mask
    for mode in ("topk", "fixed", "random"):
        for r in (0.75, 0.5):
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), SP.sparse(mode, r, check=True):
                SP.STATE.reset_stats()
                forward_from_cache(model, kv, P, ids, amask, kv_mask, labels=labels)
                st = SP.STATE.stats()
            out[f"train_{mode}_{r}"] = dict(violations=st["violations"], access_ratio=st["access_ratio"],
                                            mean_legal=st["mean_legal"], mean_kept=st["mean_kept"])
    out["train_legal_expected"] = float(kv_mask.float().sum(1).mean())
    # exact k rule on a hand-made row set: compare kept counts with the formula
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        cap = {}
        orig = SP._sparse_sdpa

        def probe(q, k, v, attn_mask=None, **kw):
            if "n" not in cap:
                Lk = k.shape[-2]
                allowed = attn_mask[..., :P] > torch.finfo(attn_mask.dtype).min / 2
                cap["n"] = allowed.expand(q.shape[0], q.shape[1], q.shape[2], P).sum(-1)
                s = torch.matmul(q.float(), k.float().transpose(-1, -2))
                cap["keep_ref"] = torch.minimum(cap["n"], torch.clamp(torch.floor(cap["n"].float() * 0.75).long(), min=1))
            return orig(q, k, v, attn_mask=attn_mask, **kw)
        SP.STATE.active = False
        torch.nn.functional.scaled_dot_product_attention = probe
        with SP.sparse("topk", 0.75):
            SP.STATE.reset_stats()
            forward_from_cache(model, kv, P, ids, amask, kv_mask, labels=labels)
        SP.install()
        n_layers = len(kv)
        out["k_rule_sum_expected_per_layer"] = int(cap["keep_ref"].sum())
        out["k_rule_sum_observed_per_layer"] = SP.STATE.sum_k / n_layers
    # causal order: perturbing a later current-input token must not change earlier-position logits
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16), SP.sparse("topk", 0.75):
        ids2 = ids.clone()
        L = int(amask[0].sum())
        ids2[0, L - 1] = (ids2[0, L - 1] + 1) % 50257
        a = forward_from_cache(model, kv, P, ids, amask, kv_mask).logits[0, :L - 1].float()
        b = forward_from_cache(model, kv, P, ids2, amask, kv_mask).logits[0, :L - 1].float()
        out["causal_maxabs_change_earlier_logits"] = float((a - b).abs().max())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jobs")
    ap.add_argument("ids", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    jobs = {j["id"]: j for j in map(json.loads, open(a.jobs))}
    model, tok = load_model()
    res = []
    for i in a.ids:
        r = check_job(jobs[i], model, tok)
        print(json.dumps(r), flush=True)
        res.append(r)
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
