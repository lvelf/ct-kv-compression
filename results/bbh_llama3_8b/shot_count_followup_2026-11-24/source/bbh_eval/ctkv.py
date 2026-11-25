"""Task-level CT-KV for Llama in Transformers 4.47.1. Implemented from the paper."""
import math
import time
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers.cache_utils import DynamicCache


@dataclass
class Prefix:
    layers: tuple            # (key, value) per layer: [1, native KV heads, positions, dim]
    original_length: int     # next RoPE position; independent of retained cache length


def _clock(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _tensor(values, device):
    return torch.tensor(values, device=device, dtype=torch.long)


def _cache(layers, batch):
    # The decoder mutates a DynamicCache. Build a new wrapper for each forward.
    return DynamicCache.from_legacy_cache(tuple(
        (k.expand(batch, -1, -1, -1), v.expand(batch, -1, -1, -1)) for k, v in layers
    ))


def _positions(batch, length, offset, device):
    return torch.arange(offset, offset + length, device=device).expand(batch, -1)


def _targets(demo, terminators):
    ids, question_length = demo[:2]
    end = demo[2] if len(demo) == 3 else len(ids)
    if len(demo) == 2:
        while end > question_length and ids[end - 1] in terminators:
            end -= 1
    if not 1 <= question_length < end <= len(ids):
        raise ValueError("Malformed demonstration token boundaries.")
    return ids, question_length, end


def _forward(model, layers, original_length, ids, mask, attentions=False):
    return model.model(
        input_ids=ids,
        attention_mask=mask,
        past_key_values=_cache(layers, ids.shape[0]),
        position_ids=_positions(ids.shape[0], ids.shape[1], original_length, ids.device),
        use_cache=True,
        output_attentions=attentions,
        return_dict=True,
    )


@torch.no_grad()
def _attention_scores(model, layers, owner, demos, terminators):
    """Mean over layers, heads, and answer prediction positions, equal weight per demo."""
    total = torch.zeros(owner.shape, dtype=torch.float32, device=owner.device)
    prefix_len = len(owner)
    for i, demo in enumerate(demos):
        ids, q, e = _targets(demo, terminators)
        # Final target is predicted by token e-2. Its identity never enters its own attention.
        ids = _tensor(ids[:e-1], owner.device).unsqueeze(0)
        mask = torch.cat([owner != i, torch.ones_like(ids[0], dtype=torch.bool)]).unsqueeze(0)
        out = _forward(model, layers, prefix_len, ids, mask, attentions=True)
        if not out.attentions or out.attentions[0] is None:
            raise ValueError("Answer attention requires SDPA with eager fallback, or eager attention.")
        for layer in out.attentions:
            total.add_(layer[0, :, q-1:e-1, :prefix_len].float().mean(dim=(0, 1)))
        del out, layer
    return total.div_(len(layers) * len(demos))


def _select(model, layers, owner, method, budget, selection_seed, demos, terminators):
    device = owner.device
    instruction = (owner == -1).nonzero().flatten()
    demo_positions = (owner >= 0).nonzero().flatten()
    n = len(demo_positions)
    keep = max(1, math.floor(budget * n))
    if keep == n:
        return torch.arange(len(owner), device=device)
    if method == "random":
        rng = torch.Generator().manual_seed(selection_seed)
        ids = torch.randperm(n, generator=rng)[:keep].to(device)
    elif method in {"answer_attention", "value_norm"}:
        if method == "answer_attention":
            scores = _attention_scores(model, layers, owner, demos, terminators)
        else:
            scores = torch.zeros_like(owner, dtype=torch.float32)
            for _, value in layers:
                scores.add_(torch.linalg.vector_norm(value[0].float(), dim=-1).mean(dim=0))
            scores.div_(len(layers))
        ids = torch.argsort(scores[demo_positions], descending=True, stable=True)[:keep]
    else:
        raise ValueError("Use full at budget 1, random, answer_attention, or value_norm.")
    return torch.cat([instruction, demo_positions[ids].sort().values])


def _loss(model, prefix, owner, group, demos, terminators, rng):
    device = owner.device
    parts = [_targets(demos[i], terminators) for i in group]
    longest = max(e-1 for _, _, e in parts)
    inputs = torch.zeros(len(group), longest, dtype=torch.long, device=device)
    query_mask = torch.zeros_like(inputs, dtype=torch.bool)
    select = torch.zeros_like(query_mask)
    target = []
    for b, (tokens, q, e) in enumerate(parts):
        inputs[b, :e-1] = _tensor(tokens[:e-1], device)
        query_mask[b, :e-1] = True
        select[b, q-1:e-1] = True
        target.extend(tokens[q:e])
    # Leave out the *entire* held-out demo's retained KVs, across every layer.
    visible = owner.unsqueeze(0) != _tensor(group, device).unsqueeze(1)
    visible &= torch.rand(visible.shape, generator=rng).to(device) >= 0.10
    mask = torch.cat([visible, query_mask], dim=1)
    hidden = _forward(model, prefix.layers, prefix.original_length, inputs, mask).last_hidden_state
    logits = model.lm_head(hidden[select])
    return F.cross_entropy(logits.float(), _tensor(target, device), reduction="sum")


def fit(model, instruction_ids, demos, method="full", budget=1.0, optimizer_seed=0,
        selection_seed=0, steps=16, microbatch=2, eos_ids=(), newline_ids=()):
    """Initialize, select, and optionally train a single shared cache from demonstration labels only."""
    if method not in {"full", "random", "answer_attention", "value_norm"} or not 0 < budget <= 1:
        raise ValueError("Invalid CT-KV method or budget.")
    if method == "full" and budget != 1:
        raise ValueError("Full CT-KV uses budget 1.")
    if not demos or not instruction_ids or steps < 0 or not 1 <= microbatch <= len(demos):
        raise ValueError("Invalid demonstration set, step count or microbatch size.")
    terminators = set(eos_ids) | set(newline_ids)
    parts = [_targets(demo, terminators) for demo in demos]
    ids = list(instruction_ids)
    owner = [-1] * len(ids)
    for i, (tokens, _, _) in enumerate(parts):
        ids.extend(tokens)
        owner.extend([i] * len(tokens))
    if steps or (budget < 1 and method == "answer_attention"):
        if len(ids) + max(e-1 for _, _, e in parts) > model.config.max_position_embeddings:
            raise ValueError("Full prefix plus a training demonstration exceeds context. No truncation.")
    if len(ids) > model.config.max_position_embeddings:
        raise ValueError("Demonstration prefix exceeds model context.")
    model.eval().requires_grad_(False)
    device = next(model.parameters()).device
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    start = _clock(device)
    owner = _tensor(owner, device)
    with torch.no_grad():
        out = model.model(input_ids=_tensor(ids, device)[None, :], use_cache=True)
        layers = tuple((k.detach(), v.detach()) for k, v in out.past_key_values)
        del out
        init_time = _clock(device)
        indices = _select(model, layers, owner, method, budget, selection_seed, demos, terminators)
        if len(indices) < len(ids):
            layers = tuple((k.index_select(2, indices), v.index_select(2, indices)) for k, v in layers)
            owner = owner[indices]
    selected_time = _clock(device)
    prefix = Prefix(layers, len(ids))
    params = [x for layer in layers for x in layer]
    losses = []
    if steps:
        for param in params:
            param.requires_grad_(True)
        optimizer = torch.optim.Adam(params, lr=0.001, foreach=False)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps)
        rng = torch.Generator().manual_seed(optimizer_seed)
        labels_per_step = sum(e-q for _, q, e in parts)
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            order = torch.randperm(len(demos), generator=rng).tolist()
            total = torch.zeros((), dtype=torch.float32, device=device)
            for begin in range(0, len(demos), microbatch):
                loss = _loss(model, prefix, owner, order[begin:begin+microbatch], demos, terminators, rng)
                loss = loss / labels_per_step
                loss.backward()
                total += loss.detach()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            scheduler.step()
            losses.append(total.item())
        optimizer.zero_grad(set_to_none=True)
        del optimizer, scheduler
        for param in params:
            param.requires_grad_(False)
        if not all(math.isfinite(value) for value in losses):
            raise ValueError("Non-finite adaptation loss.")
    finish = _clock(device)
    peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
    n_instruction = len(instruction_ids)
    kept = [i - n_instruction for i in indices.tolist() if i >= n_instruction]
    meta = dict(method=method, budget=budget, steps=steps, microbatch=microbatch,
                optimizer_seed=optimizer_seed, selection_seed=selection_seed,
                n_demo_tokens=len(ids)-n_instruction, n_instruction_tokens=n_instruction,
                n_retained_demo_tokens=len(kept), retained_demo_indices=kept,
                retained_cache_bytes=sum(p.numel()*p.element_size() for p in params),
                adaptation_seconds=finish-start, adaptation_peak_allocated_bytes=peak,
                prefix_init_seconds=init_time-start, selection_seconds=selected_time-init_time,
                training_seconds=finish-selected_time, loss_per_step=losses)
    return prefix, meta


@torch.inference_mode()
def decode(model, tokenizer, prefix, query_ids, cap):
    """Greedy answer, preserving RoPE positions even if cache positions were removed."""
    if cap < 1 or not query_ids:
        raise ValueError("A nonempty query and positive generation budget are required.")
    if prefix.original_length + len(query_ids) + cap > model.config.max_position_embeddings:
        raise ValueError("Prefix plus query and answer budget exceed model context.")
    device = next(model.parameters()).device
    cache = _cache(prefix.layers, 1)
    ids = _tensor(query_ids, device)[None, :]
    absolute = prefix.original_length
    stop_ids = model.generation_config.eos_token_id
    eos = set(stop_ids if isinstance(stop_ids, (tuple, list)) else [stop_ids])
    eos.add(tokenizer.eos_token_id)
    new_tokens = []
    for _ in range(cap):
        out = model.model(input_ids=ids, past_key_values=cache, use_cache=True,
                          position_ids=_positions(1, ids.shape[1], absolute, device))
        cache = out.past_key_values
        token = model.lm_head(out.last_hidden_state[:, -1, :]).argmax(-1).item()
        new_tokens.append(token)
        text = tokenizer.decode(new_tokens, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)
        if token in eos or "\n" in text:
            break
        absolute += ids.shape[1]
        ids = _tensor([[token]], device)
    return text, new_tokens
