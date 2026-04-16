# Data Ablation: Next Experimental Directions

Context: v1 + v2 ablation sweeps (14 + 5 runs) produced mean task deltas of ±0.005,
which is at or below single-seed noise. Hand-crafted filters on the 1.1M OSM sampling
corpus are not producing a clear signal. These are the five alternative directions
worth pursuing before running more filter sweeps.


## 1. Evaluate the corpus *without* training

Embedding-space coverage as a training-free proxy for dataset quality.

- Embed the full corpus and each eval dataset with an existing foundation model
  (CLIP, DINOv2, or Helios itself) - one forward pass, no backprop.
- For each eval sample, compute k-NN distance to the corpus embeddings.
- A candidate filter/subset is "good for task X" iff it increases k-NN coverage
  of task X's embedding distribution (lower mean distance, fewer uncovered samples).
- Ranks 50 filter candidates in hours instead of weeks.

Deliverable: per-(filter, eval_task) coverage score. Only launch GPU runs for the
top 2-3 candidates per task cluster.

Rough cost: 1 GPU-day of embedding + a few CPU hours of k-NN.


## 2. Run ablations at nano scale

Base-scale pretraining at 400k steps is an expensive estimator for a small quantity.

- Rerun the v2 sweep at `nano` (`scripts/official/nano.py`) with identical filters.
- If filter rankings reproduce: use nano as the ablation vehicle going forward.
  Order-of-magnitude cheaper per comparison.
- If rankings do *not* reproduce: the ordering is unstable at base scale too.
  Stop chasing the small deltas and move on to #3 or #4.

Either outcome is informative. Pair with seed replicates (2-3 seeds/condition) so
you can actually estimate variance.

Rough cost: a nano sweep of 7 filters at 3 seeds ≈ 1-2 days on 8 GPUs total.


## 3. Replace hand-crafted filters with DoReMi-style mixture weights

144 features × hand-picked thresholds is underfitting the mixing problem.

- Cluster the corpus in embedding space into k = 50-200 clusters.
- Parametrize a mixture weight per cluster.
- Train a small proxy model with the cluster-mixture weights as *learnable*
  parameters, optimized against a reference model's loss (standard DoReMi).
- The resulting weights directly say which clusters to upweight - no thresholds
  to tune, no 144-feature slicing.

Directly replaces the whole filter sweep with one optimization run.
Deliverable: a weighted sampler configuration usable in production pretraining.

Rough cost: ~1 day on 8 GPUs.

References: DoReMi (Xie et al. 2023), RegMix, DataComp.


## 4. Scaling-law view, not single-point comparison

The right question isn't "which filter wins at 400k steps?" - it's "which filter
has a better data-efficiency *slope*?".

- Pick 2 candidate filters + random baseline.
- Train each at 3 data scales (e.g. 50k / 200k / 1M samples, matched epochs).
- Fit a scaling law to eval vs data.
- A genuinely better dataset has a better slope. A filter that is "0.005 better
  at 400k" is noise and will not matter at scale.

This also tells you whether your single-point comparisons are even in a regime
where filtering matters, or whether you're compute-bound (adding steps > adding
better samples).

Rough cost: 9 runs instead of 5, but answers a fundamentally different question.


## 5. Dedup first, filter second

Dedup is the single largest data-quality lever in modern pretraining across LLMs,
vision, and increasingly geospatial work. It's almost certainly live in this
corpus: near-duplicate spatial tiles across adjacent timesteps and overlapping
windows.

- Hash-based near-dup detection on the 144-feature vectors or a cheap embedding.
- Cluster + cap samples-per-cluster (e.g. max K samples per
  `100km × 100km × season` cell).
- Compare dedup'd pool vs raw pool at matched budget.

Strong prior of working without any filter tuning. No tradeoff dynamics to
navigate (unlike hard filters, which always hurt some task).

Rough cost: hours to produce, one training run to validate.


---

## Priority ordering

If picking in order of expected payoff / cost:

1. **#5 (dedup)** - highest prior of working, cheapest to implement, lit-backed.
2. **#1 (embedding coverage)** - turns the 50-filter search into a CPU job.
3. **#2 (nano scale)** - makes all future ablations cheaper + reveals if rankings
   are even stable.
4. **#3 (DoReMi weights)** - the principled replacement for hand-crafted filters.
5. **#4 (scaling laws)** - expensive but definitive; save for the 2-3 candidates
   that survive #1-#3.

Common thread: either make the estimator cheaper (#1, #2) or change the quantity
being estimated to something with a larger effect size (#3, #4, #5). The current
setup is a very expensive estimator of a very small quantity.
