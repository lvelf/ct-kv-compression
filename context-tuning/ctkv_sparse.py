"""Stage 2: per-query sparse reading of the demonstration KV prefix (exact top-k, diagnostic implementation).

Installed by wrapping torch.nn.functional.scaled_dot_product_attention. Both Llama (LlamaSdpaAttention) and GPT-2
(GPT2SdpaAttention) call it after RoPE / head split and, for Llama, after repeat_kv, so the wrapper sees one row of
query-key scores per (batch, query head, query token) with the model's own scaling, positions, GQA mapping and 4D
mask. When no sparse context is active the original kernel runs unchanged (production dense path).

Inside an active context, for every attention call:
  scores = Q K^T * scale (+ the model's mask), in float32
  legal  = demonstration columns (the first P key positions with demo_cols True) that the original mask allows
           (leave-one-out and token-dropout masking happen before the selection: they are already in the mask)
  n      = legal positions in this row;  k = min(n, max(1, floor(r * n)));  n = 0 -> no selection
  keep   = the k legal positions with the highest ranking key
           topk:   the row's own query-key score            (dynamic, recomputed every forward)
           fixed:  a fixed per-position score (e.g. Ryan's answer_attention), same legal set and k
           random: i.i.d. uniform keys from a seeded CUDA generator, same legal set and k
           dense:  keep every legal position (r = 1 equivalence / diagnostic dense through the same code)
  then one joint softmax over kept demo positions + instruction + current-input positions (original causal mask),
  dropped demo positions get -inf, so no probability and no gradient flow through them. The ranking itself is
  not differentiated; gradients flow along the selected edges into Q, K, V (and so into the trainable cache).
Columns >= P (the current input / generated tokens) and instruction columns (demo_cols False) are never selected
against. Full logits are materialised, so this measures prediction quality, not speed.
"""
import math
from contextlib import contextmanager

import torch
import torch.nn.functional as F

_ORIG_SDPA = F.scaled_dot_product_attention


class SparseState:
    def __init__(self):
        self.active = False
        self.mode = "dense"
        self.r = 1.0
        self.P = 0
        self.demo_cols = None       # [P] bool
        self.owner = None           # [P] long: demonstration index, -1 for instruction
        self.n_demos = 0
        self.fixed_score = None     # [P] float
        self.gen = None             # CUDA generator for random
        self.check = False          # verify zero probability on non-selected demo columns
        self.track = False          # accumulate kept / legal counts per demonstration
        self.reset_stats()

    def reset_stats(self):
        self.calls = 0
        self.rows = 0
        self.sum_n = 0.0
        self.sum_k = 0.0
        self.violations = 0
        self.kept_by_demo = None
        self.legal_by_demo = None

    def stats(self):
        return dict(calls=self.calls, rows=self.rows, mean_legal=self.sum_n / max(self.rows, 1),
                    mean_kept=self.sum_k / max(self.rows, 1),
                    access_ratio=self.sum_k / self.sum_n if self.sum_n else float("nan"),
                    violations=int(self.violations))


STATE = SparseState()


def configure(P, demo_cols, owner=None, n_demos=0, fixed_score=None, mask_seed=None, device="cuda"):
    STATE.P = int(P)
    STATE.demo_cols = demo_cols.to(device).bool()
    STATE.owner = None if owner is None else owner.to(device).long()
    STATE.n_demos = n_demos
    STATE.fixed_score = None if fixed_score is None else fixed_score.to(device).float()
    STATE.gen = None
    if mask_seed is not None:
        STATE.gen = torch.Generator(device=device)
        STATE.gen.manual_seed(int(mask_seed))


def reseed(mask_seed):
    if STATE.gen is not None:
        STATE.gen.manual_seed(int(mask_seed))


@contextmanager
def sparse(mode, r=1.0, check=False, track=False):
    """Activate sparse reading of the prefix for the enclosed forwards; mode 'off' leaves the production kernel."""
    if mode in (None, "off", "prod"):
        yield STATE
        return
    if mode not in ("dense", "topk", "fixed", "random"):
        raise ValueError(mode)
    if mode == "fixed" and STATE.fixed_score is None:
        raise ValueError("fixed mode needs configure(fixed_score=...)")
    if mode == "random" and STATE.gen is None:
        raise ValueError("random mode needs configure(mask_seed=...)")
    prev = (STATE.active, STATE.mode, STATE.r, STATE.check, STATE.track)
    STATE.active, STATE.mode, STATE.r, STATE.check, STATE.track = True, mode, float(r), check, track
    try:
        yield STATE
    finally:
        STATE.active, STATE.mode, STATE.r, STATE.check, STATE.track = prev


def _sparse_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **kw):
    if not STATE.active or kw.get("enable_gqa"):
        return _ORIG_SDPA(query, key, value, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=is_causal,
                          scale=scale, **kw)
    if dropout_p:
        raise ValueError("attention dropout is not supported in the sparse path")
    Lq, Lk, D = query.shape[-2], key.shape[-2], query.shape[-1]
    scale = 1.0 / math.sqrt(D) if scale is None else scale
    s = torch.matmul(query.float(), key.float().transpose(-1, -2)) * scale          # [B, H, Lq, Lk]
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            allowed = attn_mask
            s = s.masked_fill(~attn_mask, float("-inf"))
        else:
            s = s + attn_mask.float()
            allowed = attn_mask > torch.finfo(attn_mask.dtype).min / 2
    elif is_causal:
        allowed = torch.ones(Lq, Lk, dtype=torch.bool, device=s.device).tril(diagonal=Lk - Lq)
        s = s.masked_fill(~allowed, float("-inf"))
    else:
        allowed = torch.ones(1, 1, 1, Lk, dtype=torch.bool, device=s.device)
    P = min(STATE.P, Lk)
    if P > 0:
        legal = allowed[..., :P].expand(*s.shape[:-1], P) & STATE.demo_cols[:P]
        n = legal.sum(-1)                                                            # [B, H, Lq]
        if STATE.mode == "dense":
            keep = legal
        else:
            k = torch.minimum(n, torch.clamp(torch.floor(n.float() * STATE.r).long(), min=1))
            if STATE.mode == "topk":
                rank_key = s[..., :P].detach()
            elif STATE.mode == "fixed":
                rank_key = STATE.fixed_score[:P].expand(*s.shape[:-1], P)
            else:
                rank_key = torch.rand(s.shape[:-1] + (P,), generator=STATE.gen, device=s.device)
            rank_key = rank_key.masked_fill(~legal, float("-inf"))
            order = rank_key.argsort(dim=-1, descending=True, stable=True)
            rank = torch.empty_like(order)
            rank.scatter_(-1, order, torch.arange(P, device=s.device).expand_as(order))
            keep = (rank < k.unsqueeze(-1)) & legal
            s = torch.cat([s[..., :P].masked_fill(legal & ~keep, float("-inf")), s[..., P:]], dim=-1)
        STATE.calls += 1
        STATE.rows += n.numel()
        STATE.sum_n += float(n.sum())
        STATE.sum_k += float(keep.sum())
        if STATE.track and STATE.owner is not None:
            kept_pos = keep.sum(dim=tuple(range(keep.dim() - 1))).float()
            legal_pos = legal.sum(dim=tuple(range(legal.dim() - 1))).float()
            idx = STATE.owner[:P].clamp(min=0)
            isdemo = (STATE.owner[:P] >= 0).float()
            kb = torch.zeros(STATE.n_demos, device=s.device).index_add_(0, idx, kept_pos * isdemo)
            lb = torch.zeros(STATE.n_demos, device=s.device).index_add_(0, idx, legal_pos * isdemo)
            STATE.kept_by_demo = kb if STATE.kept_by_demo is None else STATE.kept_by_demo + kb
            STATE.legal_by_demo = lb if STATE.legal_by_demo is None else STATE.legal_by_demo + lb
    p = torch.softmax(s, dim=-1)
    p = torch.nan_to_num(p, nan=0.0)  # rows with every key masked (never happens with a visible current token)
    if STATE.check and P > 0 and STATE.demo_cols is not None:
        bad = (p[..., :P] > 0) & ~keep & STATE.demo_cols[:P]   # instruction columns are never selectable
        STATE.violations += int(bad.sum())
    out = torch.matmul(p, value.float())
    return out.to(query.dtype)


def install():
    """Route SDPA calls through the sparse wrapper (a no-op unless a sparse() context is active)."""
    F.scaled_dot_product_attention = _sparse_sdpa
    torch.nn.functional.scaled_dot_product_attention = _sparse_sdpa


def uninstall():
    F.scaled_dot_product_attention = _ORIG_SDPA
    torch.nn.functional.scaled_dot_product_attention = _ORIG_SDPA
