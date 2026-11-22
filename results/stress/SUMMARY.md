# Stress tests of CT-KV on NLP-LR (GPT-2 Large): summary

All runs: official code with our added flags, RTX 5090, bf16, 21 tasks per split, 200 CT-KV epochs unless noted.
Detailed tables: `a3_epochs/report.md` (epochs, K/V diagnostics), `b3_compress/report.md` (compression, memory, time),
`extra_report.md` (noise, tune only K or V). Regenerate with `analyze_a3.py`, `analyze_b3.py`, `analyze_extra.py`.

## Noise floor (A4)

- Same seed, 4 runs on split 87: average 44.88 ± 0.06 (bf16 GPU nondeterminism); single tasks move up to 0.5.
- Different seeds (6 runs): 44.73 ± 0.24; single tasks move up to 2 points (glue-wnli).
- So differences under ~0.5 on one split, or ~0.3 on a 3-split mean, are noise.

## Epochs (A3, B1), 5 splits

| epochs | 0 (ICL) | 25 | 50 | 100 | 200 | 400 | 800 |
|---|---|---|---|---|---|---|---|
| mean | 35.74 | 39.28 | 40.88 | 42.57 | **44.36** | 44.23 | 44.11 |

- 200 (the paper's default) is the best fixed value; more epochs degrade slowly.
- Picking the best epochs per task using test scores gives 45.44 (per task) to 46.22 (per task and split): at most ~1–2 points from tuning the schedule.

## Where CT-KV changes the cache (A5, A5b)

- V gets ~5× the gradient energy of K and drifts ~5× more (3.5% vs 0.7% relative); layers 0–2 change most.
- Tuning only V keeps almost all of the gain: 44.56 vs 44.74 (3 splits). Only K: 39.95. ICL: 35.86.
- But it saves almost no memory or time (peak 14.2 vs 14.3 GB, 9.1 vs 9.5 s per task): the cost is activations, not the KV parameters.

## Compressing the demonstration KV before CT-KV (B3), splits 13/21/87

Each demonstration keeps a fraction of its tokens (same tokens in all layers and heads), then CT-KV runs on the compressed cache.

| keep ratio | KV MB | train peak GB | random | random + sink | attn | l2 |
|---|---|---|---|---|---|---|
| 1.0 | 79 | 14.6 | 44.76 | 44.76 | 44.76 | 44.76 |
| 0.75 | 61 | 12.6 | 43.63 | 43.67 | 44.16 | 43.81 |
| 0.5 | 40 | 10.4 | 43.16 | 43.11 | 42.55 | 43.01 |
| 0.25 | 21 | 8.2 | 36.75 | 40.51 | 39.87 | 40.22 |
| 0.125 | 11 | 7.1 | 34.64 | 38.47 | 39.13 | 38.35 |

Uncompressed ICL on these splits: 35.86. "sink" = token 0 of the whole prefix.

Findings:
1. **The attention sink decides whether CT-KV works on a compressed cache.** Every method that keeps token 0
   (attn, l2, first, random_sink, random_segstart) recovers most of the gain; every method that drops it
   (random at ≤25%, last, l2_high) or averages it away (pool) ends near ICL.
   With the fixed seed, plain random happened to keep token 0 at 75% and 50%, so those rows equal random_sink.
2. **The sink only matters for training.** Without CT-KV (e0), keeping token 0 or not makes no difference
   (e.g. 25%: 32.63 random vs 32.33 random_sink).
3. **Once the sink is kept, smart selection does not beat random.** attn, l2 and random_sink are within
   about ±0.7 of each other at every budget, with no consistent winner; this is near the noise floor.
4. **Best-ranked without training ≠ best after training.** pool is among the best at e0 (≈34) but gains
   only 1–2 points from CT-KV; l2 is the worst at e0 but gains the most.
5. **Budget/accuracy trade-off.** 75% keeps ~90% of CT-KV's gain at 87% of the training peak memory; 50% keeps
   ~80% at 71%; 12.5% still beats uncompressed ICL by ~2.5–3 points at 49% of the memory.
6. **Compression saves memory, not much time.** Training time per task falls only from 10.0 s to 8.0 s at 12.5%,
   and evaluation time barely changes.

## Implications for the planned "select useful KV entries" method

- The baseline must be **random + sink** (and several random seeds), not plain random. Plain random is weak
  only because it drops token 0, and beating it says nothing about selection quality.
- Remaining headroom over random + sink is small: at 50% the gap to uncompressed CT-KV is ~1.6 points.
- GPT-2 uses absolute positions, so compression frees KV memory but cannot free context slots for more demos.
