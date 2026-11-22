import json
from collections import defaultdict
import gc
import math
import time
from datetime import timedelta
from tqdm import tqdm
from functools import partial
import argparse
from typing import Callable, List, Tuple, Dict, Any, Iterator, Optional

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from transformers import (
    AutoTokenizer,
    get_cosine_schedule_with_warmup,
    GPT2LMHeadModel,
    GPT2Config,
)
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import ProjectConfiguration, set_seed

from data_utils import (
    EvalDataset,
    GSDataset,
    collate_fn_eval,
    collate_fn_gs,
)


import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["NCCL_TIMEOUT"] = "28800" # 8 hours
os.environ["NCCL_TIMEOUT_MS"] = "28800000"
os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["NCCL_BLOCKING_WAIT"] = "1"
os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"


def get_individual_loss(lm_logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """Return the mean next-token loss of each sequence, ignoring masked labels."""
    # move labels to correct device to enable model parallelism
    labels = label_ids.to(lm_logits.device)

    # shift so that tokens < n predict n
    losses = []
    for logs, labs in zip(lm_logits, labels):
        shift_logits = logs[:-1, :].contiguous()
        shift_labels = labs[1:].contiguous()
        # flatten the tokens
        loss_fct = nn.CrossEntropyLoss(reduction='none')
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        loss = loss[labs[1:] != -100].mean()
        losses.append(loss)

    return torch.stack(losses)


def compute_macrof1_or_accuracy(predictions, groundtruths, is_classification) -> float:
    """Return macro-F1 for classification tasks and accuracy otherwise, as in MetaICL."""
    # accuracy measurement to use the same evaluation setup as MetaICL
    accs = []
    precisions = defaultdict(list)
    recalls = defaultdict(list)
    for prediction, groundtruth in zip(predictions, groundtruths):
        prediction = prediction.strip()
        groundtruth = groundtruth.strip()
        is_correct = prediction==groundtruth
        accs.append(is_correct)
        if is_classification:
            recalls[groundtruth].append(is_correct)
            precisions[prediction].append(is_correct)

    if not is_classification:
        return float(np.mean(accs))

    f1s = []
    for key in recalls:
        precision = np.mean(precisions[key]) if key in precisions else 1.0
        recall = np.mean(recalls[key])
        if precision+recall==0:
            f1s.append(0)
        else:
            f1s.append(2*precision*recall / (precision+recall))

    return float(np.mean(f1s))


def chunks(lst: List[Any], n: int) -> Iterator[List[Any]]:
    """Yield successive chunks of n items from a list."""
    # iterator for batched items from list
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


@torch.no_grad()
def compress_kv(
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    demon_start_idxs: List[int],
    method: str,
    keep_ratio: float,
    attentions: Optional[Tuple[torch.Tensor]] = None,
    seed: int = 0,
) -> Tuple[Tuple[Tuple[torch.Tensor, torch.Tensor]], List[int]]:
    """Shrink the demonstration KV cache to about keep_ratio of its tokens.

    Each demonstration segment keeps ceil(keep_ratio * len) tokens, so that every demonstration
    survives and Leave-One-Out Masking still has a segment to mask. The same tokens are kept in
    every layer and head. GPT-2 adds absolute position embeddings at the input, so kept entries
    keep their original position information; new tokens still start at the uncompressed length.

    Methods:
      l2      keep the tokens with the lowest key L2 norm, averaged over layers and heads (Devoto et al., 2024)
      l2_high keep the tokens with the highest key norm (control for l2)
      attn    keep the tokens that receive the most attention, averaged over layers, heads and the
              queries that can see them (H2O-style, normalized for causal masking)
      random  keep random tokens (control)
      random_sink      random, but always keep token 0 of the whole prefix (the attention sink)
      random_segstart  random, but always keep the first token of every demonstration
      first   keep the first tokens of each segment (the input side of the pair)
      last    keep the last tokens of each segment (the output side of the pair)
      pool    split each segment into equal contiguous chunks and average K and V within each chunk

    Returns the compressed cache, the start index of each demonstration in the compressed cache, and
    stats about which tokens were kept (None for pool).
    """
    seq_len = past_key_values[0][0].shape[2]
    bounds = list(demon_start_idxs) + [seq_len]
    segments = [(bounds[i], bounds[i + 1]) for i in range(len(demon_start_idxs))]

    if method == "pool":
        new_k = [[] for _ in past_key_values]
        new_v = [[] for _ in past_key_values]
        new_starts, cur = [], 0
        for start, end in segments:
            n = end - start
            m = max(1, math.ceil(keep_ratio * n))
            chunk_edges = np.linspace(start, end, m + 1).round().astype(int)
            new_starts.append(cur)
            for a, b in zip(chunk_edges[:-1], chunk_edges[1:]):
                if b <= a:
                    continue
                for l, (k, v) in enumerate(past_key_values):
                    new_k[l].append(k[:, :, a:b].float().mean(dim=2, keepdim=True).to(k.dtype))
                    new_v[l].append(v[:, :, a:b].float().mean(dim=2, keepdim=True).to(v.dtype))
                cur += 1
        compressed = tuple((torch.cat(ks, dim=2), torch.cat(vs, dim=2)) for ks, vs in zip(new_k, new_v))
        return compressed, new_starts, None

    # token scores: higher = keep
    device = past_key_values[0][0].device
    if method in ("l2", "l2_high"):
        norms = torch.stack([k[0].float().norm(dim=-1).mean(dim=0) for k, _ in past_key_values]).mean(dim=0)  # [seq]
        scores = -norms if method == "l2" else norms
    elif method == "attn":
        assert attentions is not None, "attn compression needs output_attentions"
        received = torch.stack([a[0].float().sum(dim=1).mean(dim=0) for a in attentions]).mean(dim=0)  # [seq], summed over queries
        n_queries = torch.arange(seq_len, 0, -1, device=received.device, dtype=torch.float)  # token j is seen by seq_len - j queries
        scores = received / n_queries
    elif method in ("random", "random_sink", "random_segstart"):
        gen = torch.Generator().manual_seed(seed)
        scores = torch.rand(seq_len, generator=gen).to(device)
        if method == "random_sink":
            scores[0] = float("inf")
        elif method == "random_segstart":
            for start, _ in segments:
                scores[start] = float("inf")
    elif method == "first":
        scores = -torch.arange(seq_len, device=device, dtype=torch.float)
    elif method == "last":
        scores = torch.arange(seq_len, device=device, dtype=torch.float)
    else:
        raise ValueError(f"unknown kv compression method {method}")

    keep, new_starts, rel_pos = [], [], []
    for start, end in segments:
        n = end - start
        m = max(1, math.ceil(keep_ratio * n))
        top = sorted((torch.topk(scores[start:end], m).indices + start).tolist())  # keep original order
        new_starts.append(len(keep))
        keep.extend(top)
        rel_pos.extend((j - start) / max(n - 1, 1) for j in top)
    keep_set = set(keep)
    info = {
        "kept_token0": 0 in keep_set,
        "kept_segstart_frac": sum(s in keep_set for s, _ in segments) / len(segments),
        "kept_mean_rel_pos": float(np.mean(rel_pos)),  # 0 = segment start, 1 = segment end
    }
    idx = torch.tensor(keep, device=device)
    compressed = tuple((k.index_select(2, idx), v.index_select(2, idx)) for k, v in past_key_values)
    return compressed, new_starts, info


@torch.no_grad()
def inference_time_optimization(
    model: nn.Module,
    dataset: EvalDataset,
    accelerator: Accelerator,
    batch_size: int,
    collate_fn: Callable,
    log_every: int,
    zero_shot: bool,
    epochs: int,
    ct_batch_size: int,
    lr: float,
    token_dropout: float,
    diag_every: int = 0,
    kv_compress: str = "none",
    kv_keep_ratio: float = 1.0,
    compress_seed: int = 0,
    train_kv: str = "kv",
) -> Dict[str, Any]:
    """Evaluate every task, optionally refining its demonstration KV cache with context tuning first.

    Returns a dict with per-task scores, the average score, and per-task diagnostics
    (training time, peak memory, prefix length, and CT-KV training statistics).
    """
    model.eval()

    embed_tokens = model.transformer.wte
    model.generation_config.pad_token_id = dataset.tokenizer.pad_token_id

    # outputs
    output_list = []
    task_to_diag = {}

    for task in dataset.tasks:
        print(f"processing {task}...")
        model.eval()
        diag = {}
        task_name = task  # `task` is overwritten by the batch's task list in the eval loop below
        torch.cuda.reset_peak_memory_stats()

        # get demonstration ids and indices
        demon_input_ids = None
        demon_start_idxs = None
        for data in dataset.data:
            if data['task'] == task:
                demon_input_ids = data["demon_input_ids"].unsqueeze(0).to(accelerator.device)
                demon_start_idxs = data["demon_start_idxs"]
                break

        # initialize kv
        with accelerator.autocast():
            init_out = model(
                input_ids=demon_input_ids,
                output_hidden_states=True,
                output_attentions=(kv_compress == "attn"),
            )
            past_key_values = init_out.past_key_values
        diag["prefix_len"] = int(demon_input_ids.shape[1])
        diag["num_demonstrations"] = len(demon_start_idxs)

        # optionally compress the initial kv before (optional) context tuning
        kv_start_idxs = demon_start_idxs
        if kv_compress != "none":
            past_key_values, kv_start_idxs, keep_info = compress_kv(
                past_key_values=past_key_values,
                demon_start_idxs=demon_start_idxs,
                method=kv_compress,
                keep_ratio=kv_keep_ratio,
                attentions=init_out.attentions,
                seed=compress_seed,
            )
            if keep_info is not None:
                diag.update(keep_info)
        diag["kv_len"] = int(past_key_values[0][0].shape[2])
        del init_out
        torch.cuda.empty_cache()
        gc.collect()

        # use ctkv to refine kv
        if epochs > 0:
            torch.cuda.synchronize()
            train_start = time.time()
            with accelerator.no_sync(model):
                past_key_values = tuple(
                    (layer_k.to(torch.bfloat16), layer_v.to(torch.bfloat16))
                    for layer_k, layer_v in past_key_values
                )
                model, past_key_values, train_stats = context_tuning(
                    demonstration_pairs=dataset.task_to_demonstrations[task],
                    eval_dataset=dataset,
                    accelerator=accelerator,
                    model=model,
                    demon_start_idxs=kv_start_idxs,
                    past_key_values=past_key_values,
                    demon_input_ids_len=demon_input_ids.shape[1],
                    epochs=epochs,
                    lr=lr,
                    batch_size=ct_batch_size,
                    token_dropout=token_dropout,
                    diag_every=diag_every,
                    train_kv=train_kv,
                )
                torch.cuda.synchronize()
                diag["train_time_s"] = time.time() - train_start
                diag["train_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
                diag.update(train_stats)
                torch.cuda.empty_cache()
                gc.collect()

        torch.cuda.reset_peak_memory_stats()
        eval_start = time.time()
        process_data_idxs = [i for i, d in enumerate(dataset.data) if d['task'] == task]
        data_idxs = [idxs for idxs in chunks(process_data_idxs, batch_size)]
        progress_bar = tqdm(
            range(len(data_idxs)),
            desc=f"evaluating on {task}...",
            disable=not accelerator.is_local_main_process,
        )
        for eval_step, batch_idxs in enumerate(data_idxs):
            batch_data = [dataset[i] for i in batch_idxs]
            bs = len(batch_data)
            batch = collate_fn(batch_data)

            # get tensors
            task = batch['task']
            test_idx = batch['test_idx']
            option = batch['option']
            correct_option = batch['correct_option']
            gen_input_ids = batch["gen_input_ids"].to(accelerator.device)
            gen_attention_mask = batch["gen_attention_mask"].to(accelerator.device)
            gen_label_ids = batch["gen_label_ids"].to(accelerator.device)

            with accelerator.autocast():
                # expand past key values
                batch_past_key_values = [
                    (
                        layer_k.detach().clone().expand(bs, *layer_k.shape[1:]),
                        layer_v.detach().clone().expand(bs, *layer_v.shape[1:]),
                    )
                    for layer_k, layer_v in past_key_values
                ]
                batch_past_key_values_attention_mask = torch.ones(
                    (bs, batch_past_key_values[0][0].shape[2]),
                    device=accelerator.device,
                    dtype=torch.int64
                )

                gen_inputs_embeds = embed_tokens(gen_input_ids)
                gen_attention_mask = torch.cat([batch_past_key_values_attention_mask, gen_attention_mask], dim=1)

                # build position ids (does NOT depend on dropout)
                attention_mask_after_kv = gen_attention_mask[:, batch_past_key_values[0][0].shape[2]:]
                position_ids = []
                position_start = demon_input_ids.shape[1]
                for mask_after_kv in attention_mask_after_kv:
                    sequence_position_ids = torch.zeros(gen_inputs_embeds.shape[1], device=accelerator.device, dtype=torch.int64)
                    n_new_positions = mask_after_kv.sum()
                    new_positions = torch.tensor(range(position_start, position_start + n_new_positions), device=accelerator.device, dtype=torch.int64)
                    sequence_position_ids[:n_new_positions] = new_positions
                    position_ids.append(sequence_position_ids)
                position_ids = torch.stack(position_ids)

                model_out = model(
                    inputs_embeds=gen_inputs_embeds,
                    attention_mask=gen_attention_mask if not zero_shot else attention_mask_after_kv,
                    past_key_values=batch_past_key_values if not zero_shot else None,
                    position_ids=position_ids if not zero_shot else None,
                )
                losses = get_individual_loss(lm_logits=model_out.logits.half(), label_ids=gen_label_ids)

            for x0, x1, x2, x3, x4 in zip(losses, task, test_idx, option, correct_option):
                output_list.append((x0.item(), x1, x2, x3, x4))
            if (eval_step + 1) % log_every == 0:
                progress_bar.update(log_every)

            torch.cuda.empty_cache()
            gc.collect()

        diag["eval_time_s"] = time.time() - eval_start
        diag["eval_peak_mem_mb"] = torch.cuda.max_memory_allocated() / 2**20
        task_to_diag[task_name] = diag

    # determine which tasks are classification (for macro-f1)
    task_to_is_clf = {}
    for task in dataset.tasks:
        meta_data_path = os.path.join('config/tasks', f'{task}.json')
        task_meta_data = json.load(open(meta_data_path, 'r'))
        task_to_is_clf[task] = task_meta_data['task_type'] == "classification"

    # metrics
    task_to_score = {}
    for task in dataset.tasks:
        task_outs = [x for x in output_list if x[1] == task]
        if len(task_outs) == 0:
            continue

        preds, gts = [], []
        test_idxs = set(x[2] for x in task_outs)
        for test_i in test_idxs:
            task_test_outs = [x for x in task_outs if x[2] == test_i]
            correct_option = task_test_outs[0][4]
            # choose option with lowest loss
            lowest_loss = float('inf')
            chosen_option = None
            for x in task_test_outs:
                if x[0] < lowest_loss:
                    lowest_loss = x[0]
                    chosen_option = x[3]
            # record
            preds.append(chosen_option)
            gts.append(correct_option)

        task_to_score[task] = compute_macrof1_or_accuracy(preds, gts, task_to_is_clf[task])

    # display scores
    sorted_tasks = sorted(task_to_score.keys())
    for task in sorted_tasks:
        print(f"{task} clf {task_to_is_clf[task]} has a score {task_to_score[task]}")
    score = sum(v for v in task_to_score.values()) / len(task_to_score)
    print(f'average score: {score}')

    return {
        "average_score": score,
        "task_to_score": task_to_score,
        "task_to_is_clf": task_to_is_clf,
        "task_to_diag": task_to_diag,
    }


@torch.enable_grad()
def context_tuning(
    demonstration_pairs: List[Dict],
    eval_dataset: EvalDataset,
    accelerator: Accelerator,
    model: nn.Module,
    # inputs
    demon_start_idxs: List[int],
    past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]],
    demon_input_ids_len: int,
    # config
    epochs: int,
    lr: float,
    batch_size: int,
    token_dropout: float,
    diag_every: int = 0,
    train_kv: str = "kv",
):
    """Optimize the demonstration KV cache on the demonstration pairs.

    Returns (model, tuned cache, stats). stats holds the full-batch training loss of every epoch,
    the per-layer mean squared gradient (a diagonal-Fisher proxy) of K and V at epoch 0 and averaged
    over training, and, every diag_every epochs, the per-layer relative drift ||X - X0|| / ||X0||
    of K and V from their ICL initialization.
    """
    # detach a copy of the cache
    past_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    )
    init_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    )
    num_layers = len(past_key_values)
    kv_len = past_key_values[0][0].shape[2]  # < demon_input_ids_len if the cache was compressed
    epoch_losses = []
    grad_sq_k_sum = torch.zeros(num_layers, dtype=torch.float64)
    grad_sq_v_sum = torch.zeros(num_layers, dtype=torch.float64)
    grad_sq_k_epoch0, grad_sq_v_epoch0 = None, None
    drift = {}

    def relative_drift():
        k_drift, v_drift = [], []
        for (k, v), (k0, v0) in zip(past_key_values, init_key_values):
            k_drift.append(((k.float() - k0.float()).norm() / k0.float().norm()).item())
            v_drift.append(((v.float() - v0.float()).norm() / v0.float().norm()).item())
        return {"k": k_drift, "v": v_drift}

    # get program parameters
    program_params = []

    # tune the initialized K and/or V (train_kv: "kv" = both, as in the paper; "k" or "v" = only one)
    for layer_k, layer_v in past_key_values:
        if "k" in train_kv:
            program_params.append(layer_k)
        if "v" in train_kv:
            program_params.append(layer_v)

    # dataset
    gs_dataset = GSDataset(
        demonstration_pairs={i: p for i, p in enumerate(demonstration_pairs)},
        tokenizer=eval_dataset.tokenizer,
        pad_side='right',
        past_kv_len=demon_input_ids_len,
    )

    # dataloader
    batch_size = min(batch_size, len(gs_dataset))
    gs_collate_fn = partial(collate_fn_gs, dataset=gs_dataset)
    gs_loader = DataLoader(
        gs_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=gs_collate_fn,
        drop_last=False, # full batch
        num_workers=0,
    )

    # set requires grad
    for p in program_params:
        p.requires_grad = True

    # optimizer
    optimizer_grouped_params = [{"params": program_params, "lr": lr}]
    all_params = program_params
    optim = torch.optim.AdamW(optimizer_grouped_params, weight_decay=0.0)
    optim.zero_grad()

    # lr scheduler
    scheduler = get_cosine_schedule_with_warmup(optim, num_warmup_steps=0, num_training_steps=epochs)

    # set up training
    model.train()

    module = model
    embed_tokens = module.transformer.wte

    # train!
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in gs_loader:
            pair_input_ids = batch["input_ids"].to(accelerator.device)
            pair_attention_mask = batch["attention_mask"].to(accelerator.device)
            pair_label_ids = batch["label_ids"].to(accelerator.device)
            pair_example_idx = batch["example_idx"]
            device, dtype = pair_input_ids.device, pair_input_ids.dtype
            bs = pair_input_ids.shape[0]

            batch_past_key_values_attention_mask = torch.ones((bs, past_key_values[0][0].shape[2]), device=accelerator.device, dtype=torch.int64)
            batch_past_key_values = tuple(
                (layer_k.expand(bs, -1, -1, -1), layer_v.expand(bs, -1, -1, -1)) # expand here because no modifications
                for layer_k, layer_v in past_key_values
            )

            # leave one out
            for batch_i, idx in enumerate(pair_example_idx):
                # segment boundaries in kv coordinates (differ from token positions after compression)
                start = demon_start_idxs[idx]
                end = demon_start_idxs[idx + 1] if idx < len(demon_start_idxs) - 1 else kv_len
                batch_past_key_values_attention_mask[batch_i, start:end] = 0

            # token dropout
            if token_dropout != 0.0:
                drop_mask = (torch.rand_like(batch_past_key_values_attention_mask, dtype=torch.float) > token_dropout).float()
                batch_past_key_values_attention_mask = (batch_past_key_values_attention_mask * drop_mask).long()

            with accelerator.autocast():
                position_ids = torch.zeros((bs, pair_input_ids.shape[1]), device=device, dtype=torch.int64)
                new_lens = pair_attention_mask.sum(dim=1)
                for task_position_ids, new_len in zip(position_ids, new_lens):
                    new_positions = torch.tensor(range(demon_input_ids_len, demon_input_ids_len + new_len), device=device, dtype=dtype)
                    task_position_ids[:new_len] = new_positions

                pair_inputs_embeds = embed_tokens(pair_input_ids)
                pair_attention_mask = torch.cat([batch_past_key_values_attention_mask, pair_attention_mask], dim=1)
                model_kwargs = {
                    "inputs_embeds": pair_inputs_embeds,
                    "attention_mask": pair_attention_mask,
                    "labels": pair_label_ids,
                    "use_cache": True,
                    "past_key_values": batch_past_key_values,
                    "position_ids": position_ids,
                }

                # get loss
                model_out = model(**model_kwargs)
                loss = model_out.loss * bs / batch_size

            accelerator.backward(loss)
            epoch_loss += loss.item()

        epoch_losses.append(epoch_loss)
        with torch.no_grad():
            gsq = lambda t: t.grad.float().pow(2).mean().item() if t.grad is not None else 0.0  # 0 if frozen
            gk = torch.tensor([gsq(layer_k) for layer_k, _ in past_key_values], dtype=torch.float64)
            gv = torch.tensor([gsq(layer_v) for _, layer_v in past_key_values], dtype=torch.float64)
        grad_sq_k_sum += gk
        grad_sq_v_sum += gv
        if epoch == 0:
            grad_sq_k_epoch0, grad_sq_v_epoch0 = gk.tolist(), gv.tolist()

        accelerator.clip_grad_norm_(all_params, 1.0)
        optim.step()
        scheduler.step()
        optim.zero_grad()

        if diag_every > 0 and ((epoch + 1) % diag_every == 0 or epoch + 1 == epochs):
            drift[epoch + 1] = relative_drift()

    model.eval()

    # detach a copy of the cache
    past_key_values = tuple(
        (layer_k.detach().clone(), layer_v.detach().clone())
        for layer_k, layer_v in past_key_values
    )

    stats = {
        "epoch_losses": epoch_losses,
        "grad_sq_k_epoch0": grad_sq_k_epoch0,
        "grad_sq_v_epoch0": grad_sq_v_epoch0,
        "grad_sq_k_mean": (grad_sq_k_sum / max(epochs, 1)).tolist(),
        "grad_sq_v_mean": (grad_sq_v_sum / max(epochs, 1)).tolist(),
        "drift": drift,
    }
    return model, past_key_values, stats


def main():
    """Parse arguments, load the model and evaluation data, and run inference-time optimization."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type=str, required=True)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config_file", type=str, default="./hr_to_lr.json")
    parser.add_argument("--data_dir", type=str, default="./metaicl-data/data")
    parser.add_argument("--num_demonstrations", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument('--eval_split', type=str, default='87') # other splits: 13, 21, 42, 100
    parser.add_argument('--eval_ratio', type=float, default=1.0)
    parser.add_argument('--zero_shot', action='store_true')
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ct_batch_size", type=int, default=16) # ct uses less memory, full batch
    parser.add_argument("--token_dropout", type=float, default=0.05)
    parser.add_argument("--output_json", type=str, default=None, help="write scores and diagnostics here")
    parser.add_argument("--diag_every", type=int, default=0, help="record per-layer KV drift every N CT-KV epochs (0: off)")
    parser.add_argument("--kv_compress", type=str, default="none",
                        choices=["none", "l2", "l2_high", "attn", "random", "random_sink", "random_segstart", "first", "last", "pool"],
                        help="compress the initial demonstration KV cache (see compress_kv)")
    parser.add_argument("--kv_keep_ratio", type=float, default=1.0, help="fraction of demonstration tokens to keep")
    parser.add_argument("--train_kv", type=str, default="kv", choices=["kv", "k", "v"], help="which of K and V CT-KV tunes")
    args = parser.parse_args()
    run_start = time.time()

    # Setup accelerator
    project_config = ProjectConfiguration(project_dir=args.experiment_name)
    init_process_process_kwargs = InitProcessGroupKwargs()
    init_process_process_kwargs.timeout = timedelta(seconds=28800)
    accelerator = Accelerator(
        project_config=project_config,
        kwargs_handlers=[init_process_process_kwargs],
    )
    set_seed(args.seed + accelerator.process_index)
    torch.backends.cuda.matmul.allow_tf32 = True

    # log args
    print("#### BEGIN ALL ARGUMENTS ####")
    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")
    print("#### END ALL ARGUMENTS ####\n")

    # Load tokenizers
    tokenizer = AutoTokenizer.from_pretrained('openai-community/gpt2-large', cache_dir='./.cache')
    tokenizer.pad_token = tokenizer.eos_token

    # load model, deterministic to lower inference variance
    config = GPT2Config.from_pretrained(
        'openai-community/gpt2-large',
        attn_pdrop=0.0,
        embd_pdrop=0.0,
        resid_pdrop=0.0,
        summary_first_dropout=0.0,
        _attn_implementation='sdpa',
    )
    model = GPT2LMHeadModel.from_pretrained(
        'openai-community/gpt2-large',
        config=config,
        torch_dtype=torch.bfloat16,
        cache_dir="./.cache",
    )

    # Prepare with accelerator
    model = accelerator.prepare(model)

    # freeze the model
    for p in model.parameters():
        p.requires_grad = False

    # Build evaluation dataset
    dataset = EvalDataset(
        data_dir=args.data_dir,
        config_file=args.config_file,
        seed=args.seed,
        eval_split=args.eval_split,
        tokenizer=tokenizer,
        pad_side='right',
        eval_ratio=args.eval_ratio,
        num_demonstrations=args.num_demonstrations,
    )
    collate_fn = partial(collate_fn_eval, dataset=dataset)

    # Eval Datasets
    results = inference_time_optimization(
        model=model,
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        log_every=args.log_every,
        zero_shot=args.zero_shot,
        epochs=args.epochs,
        lr=args.lr,
        ct_batch_size=args.ct_batch_size,
        token_dropout=args.token_dropout,
        diag_every=args.diag_every,
        kv_compress=args.kv_compress,
        kv_keep_ratio=args.kv_keep_ratio,
        compress_seed=args.seed,
        train_kv=args.train_kv,
    )

    if args.output_json is not None and accelerator.is_main_process:
        results["args"] = vars(args)
        results["total_time_s"] = time.time() - run_start
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        tmp_path = args.output_json + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(results, f, indent=1)
        os.replace(tmp_path, args.output_json)  # atomic: the queue treats an existing file as "done"
        print(f"wrote {args.output_json}")

    accelerator.end_training()


if __name__ == "__main__":
    main()
