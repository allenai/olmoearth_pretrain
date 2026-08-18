# Eval Metrics and Balanced Trials

Why balanced accuracy behaves the way it does on our eval suite, how AlphaEarth
Foundations' (AEF) evaluation protocol differs from ours, and the
`balanced_trial` addition that lets us measure both.

Written 2026-08-11 from an investigation into an apparent Ethiopia kNN win over
Tessera v2. Source for all AEF protocol claims: *AlphaEarth Foundations: An
embedding field model for accurate and efficient global mapping from sparse
label data* (arXiv 2507.22291v2), Table 1 and supplemental §S4.

---

## Table of Contents

1. [The finding that started this](#the-finding-that-started-this)
2. [Balanced accuracy vs macro F1](#balanced-accuracy-vs-macro-f1)
3. [Which of our current wins are metric-fragile](#which-of-our-current-wins-are-metric-fragile)
4. [AEF's protocol](#aefs-protocol)
5. [How our protocol differs](#how-our-protocol-differs)
6. [Implemented: the `balanced_trial` addition](#implemented-the-balanced_trial-addition)
7. [Open questions](#open-questions)

---

## The finding that started this

The `ethiopia_crops` cell looked like our first win over Tessera v2 on that
dataset: kNN balanced accuracy 0.5259 (cand_ndvi, S1+S2+L8 cloudless) against
Tessera v2's 0.4894, a +3.7 point margin. Under every other metric it is a loss.

| | balanced acc | accuracy | macro F1 | rare-class F1 (c1, c2) |
|---|---|---|---|---|
| Tessera v2 LP | **0.713** | 0.822 | 0.501 | 0.132, 0.357 |
| Tessera v2 kNN | 0.489 | **0.840** | 0.431 | 0.100, 0.238 |
| cand_ndvi (all+cl) LP | 0.595 | 0.654 | 0.398 | 0.042, 0.290 |
| cand_ndvi (all+cl) kNN | 0.526 | 0.733 | 0.384 | 0.062, 0.215 |

Two things to read off this table:

- **Tessera's kNN has higher raw accuracy than its own LP** (0.840 vs 0.822)
  while scoring 22 points lower on balanced accuracy. Its LP is not extracting
  more signal; it is distributing predictions more evenly across imbalanced
  classes. Class-1 F1 of 0.132 alongside high mean recall means high recall with
  very poor precision.
- **Our kNN "win" is the same mechanism pointed at us.** We beat Tessera on
  balanced accuracy while losing on accuracy *and* macro F1, with worse F1 on
  both rare classes.

Tessera v2's embedding is better than ours on `ethiopia_crops` under every metric
that accounts for precision, with both probes. There is no Ethiopia strength to
explain — there is a metric that rewards our prediction distribution.

The eval split is heavily imbalanced, which is what makes this possible: Tessera's
kNN accuracy of 0.8395 against balanced accuracy of 0.4894 is a 35-point gap, and
on a class-balanced eval set those two numbers are equal by construction.

---

## Balanced accuracy vs macro F1

Both metrics macro-average over classes, so **both weight a 30-sample class the
same as a 3,000-sample one**. They are identical on the rare-class-weighting axis.

They differ on precision:

- **Balanced accuracy** = mean of per-class *recall*. Ignores false positives
  entirely. Over-predicting a rare class buys free recall with no offsetting
  penalty.
- **Macro F1** = mean of per-class harmonic means of precision and recall.
  Charges for false positives, so it cannot be inflated by spreading predictions.

Balanced accuracy has one property macro F1 lacks: a **task-independent chance
baseline of 1/K**, which is what makes averaging across heterogeneous tasks
(3-class descals to 39-class us_trees) interpretable, and why AEF draws a
random-chance line in every figure. Macro F1's chance level depends on both
prevalence and the predicted distribution.

**Recommendation: keep balanced accuracy as the headline; report macro F1 as a
mandatory companion; do not claim a win unless both agree.** Macro F1 is not a
strict upgrade — it is argmax-dependent and weights rare classes heavily — so it
works better as a disagreement detector than as a replacement. PR-AUC is already
logged and is threshold-free if a third opinion is wanted.

Note for any future comparison against AEF's *published* F1 numbers: §S4.1 says
they choose the binarization threshold from `[0.1 … 0.9]` to maximize balanced
accuracy on the eval set and then compute all other metrics at that threshold, so
their published F1 is handicapped by a BA-tuned threshold. Macro F1 values we
compute for AEF/Tessera embeddings through our own pipeline are unaffected.

---

## Which of our current wins are metric-fragile

Audited for cand_ndvi on S1+S2+L8+SCL across the 16 shared task×probe cells.
Win counts by metric: **10/16 balanced accuracy, 11/16 macro F1, 12/16 accuracy**.
Balanced accuracy is the *harshest* metric for us, not the friendliest.

- **Solid under all three metrics:** africa (both probes), canada fine (both),
  lcmap (both), glance kNN, us_trees LP.
- **Flip toward us under accuracy/macro F1:** canada coarse kNN, glance LP.
- **Loses under everything:** descals (both probes), ethiopia LP.
- **Flips away from us — the only one:** `ethiopia_crops` kNN.

So the headline count survives the metric change; the specific "we beat Tessera on
Ethiopia" claim does not.

A second metric-sensitivity finding, on input configs rather than tasks: for
cand_ndvi, 50 of 55 config pairs rank concordantly between balanced accuracy and
macro F1, and `S1+S2+L8 cloudless` is #1 under all three metrics. But
**"adding S1 helps" does not survive**: plain S1+S2 is +0.35 under balanced
accuracy and −0.34 / −0.36 under macro F1 / accuracy. S1 pays under every metric
only inside the cloudless full stack (+0.80 / +0.63 / +0.41), where compositing
has removed cloudy looks and radar fills the resulting temporal gaps.
"Adding Landsat helps" survives every metric.

---

## AEF's protocol

### Sampling

Table 1's columns are `Max Trial Size (n)` and `Total Sample Size (n)`. **Max
trial size is per class, not total.** The balanced draw becomes the training set
and **the remainder of the dataset is the eval set**.

The rule is most likely `n_per_class = min(cap, least-populated class)`. We
briefly inferred *half* the least class from the ethiopia counts; measured data
from the first full sweep contradicts that — see [What we draw, and why it is not
exactly theirs](#what-we-draw-and-why-it-is-not-exactly-theirs) below. **We
deliberately draw less than their rule**, for a correctness reason that stands
independently of what they did.

| dataset | classes | max trial (per class) | total available |
|---|---|---|---|
| LCMAP land cover | 6 | 300 | 26,510 |
| LCMAP land use | 6 | 300 | 26,513 |
| GLanCE | 11 | 300 | 34,885 |
| US trees | 39 | 300 | 45,382 |
| Africa crop mask | 4 | 200 | 2,556 |
| Descals oil palm | 3 | 200 | 17,477 |
| Canada crops fine | 24 | 75 | 14,566 |
| Canada crops coarse | 9 | 68 | 16,079 |
| Ethiopia crops | 4 | 49 | 2,530 |

The round values (300, 200, and 150 for land-use change) are deliberate caps; the
odd ones (75, 68, 49) are the least class binding first. The caps are intentional
— the main text says max-trial "is meant to represent more realistic sparse
dataset sizes (hundreds as opposed to thousands or millions of points)." They
also run 1-shot and 10-shot variants of the same balanced structure.

#### What we draw, and why it is not exactly theirs

The first implementation drew `min(cap, least class)`. On
`ethiopia_crops_year_aligned` that gave `n_per_class = 96` where AEF's Table 1
says **49**, on a pool of 2,529 windows against their 2,530 — the same labels.
That factor of two suggested they draw *half* the least class, and the reading
was reinforced by their other two odd values (75, 68) doubling to plausible
least classes (~150, ~136).

**Measured data from the first full sweep says otherwise.** Across all eight
datasets, `min(cap, least)` reproduces Table 1's max-trial column exactly on the
five cap-bound datasets (africa/descals 200, lcmap/glance/us_trees 300), while
`min(cap, least/2)` matches none of them — it falls below the cap everywhere.
For the half reading to hold, AEF's rarest classes would have to sit far above
what proportional scaling of our pools predicts (us_trees would need ≥600 where
scaling gives 429). So AEF almost certainly used the least class directly, and
the ethiopia 96-vs-49 gap is **specific to ethiopia** — a label-mapping
difference on that dataset, not evidence about the rule.

We still draw less than that, for a reason independent of AEF:

**Drawing the whole least class removes it from the eval set.** The remainder is
everything not drawn, so a class drawn in full contributes *zero* eval rows.
`balanced_accuracy_score` then averages per-class recall over the classes that
remain — silently, and upward, because the starved class is the rarest and
usually the hardest. Ethiopia's 4-class score becomes a 3-class score. That is
wrong on its own terms whatever AEF did, and it is what `eval_classes` /
`pool_classes` now guard.

**How we actually match them: take the whole column as a budget.** Every one of
our least classes exceeds its dataset's Table 1 value (ethiopia 96>49, canada
fine 87>75, coarse 106>68, africa 318>200, descals 290>200, lcmap 588>300,
glance 467>300, us_trees 393>300), so using that column directly as the
per-class draw gives **exact training-budget parity on all eight datasets while
leaving 12-288 rarest-class rows in the eval set**. That sidesteps the question
entirely: it does not matter whether 49/75/68 are caps they chose or least
classes binding, because either way it is the budget their predictor saw.
`AEF_MAX_TRIAL_CAPS` in `all_evals.py` carries the column;
`least_class_draw_fraction` (0.9) is left as a safety net for a future dataset
whose rarest class falls below its cap, and never fires today.

The 2026-08-11 sweep ran before this, at a flat 0.5 with only the round caps
set: internally consistent (every arm drew the same budget, so its head-to-heads
are fair) but budget-matched to AEF on no dataset.

Only datasets where the cap does not bind are affected — ethiopia, canada fine
and canada coarse. Where the least class exceeds the cap (africa's 318 against a
cap of 200, and the larger sets), the draw never came close to consuming a class.

### Folds

§S4.3: `k = 1000 / (2 · log₁₀ c′)`. Folds go *up* as `c′` gets *smaller*, and
only logarithmically:

| c′ | k |
|---|---|
| 10 | 500 |
| 49 (ethiopia) | 296 |
| 68 (canada coarse) | **273** |
| 100 | 250 |
| 200 (africa, descals) | 217 |
| 300 (the capped datasets) | 202 |

**`c′` is the per-class trial size, not the least-populated class.** The prose
calls it "the least-present class", and the two coincide whenever the trial
size is set by the least class — which is why the wording reads unambiguously
until a cap or a draw fraction binds. What settles it is that every published
`k` is reproduced by Table 1's max-trial column and by nothing else: 49 → 296,
68 → 273, 300 → 202. We pass `n_per_class`, which reproduces all four exactly.

`k = 1` when a draw consumes every class (only one balanced draw exists, so
there is no sampling variance); the error bar then comes from bootstrapping the
eval split B=100 times (§S4.4), which we do not implement — under a draw
fraction below 1.0 no dataset of ours reaches that case.

> The §S4.3 prose says Canada coarse has `c′ = 75` and `k = 273`, but 75 gives 267
> while **68 gives exactly 273**, and 68 is Table 1's value for coarse (75 is
> *fine*). The prose has the typo; Table 1 and the formula agree — and this is
> also the cleanest evidence that `c′` tracks the trial column.

The formula never leaves the 200–500 band across any plausible `c′`, so it is less
a scaling law than an engineered "do a couple hundred draws."

### Predictors

§S4.1: a scikit-learn `RidgeClassifier` with **λ = 0** — one-vs-rest ordinary
least squares with {−1, +1} targets, argmax at inference — plus kNN. They chose
these because they "require minimal parameterization which avoids unduly
penalizing any given method due to non-optimal hyperparameters."

**There is no model selection step**, which is what makes 200–500 refits per task
affordable: each is a closed-form solve, not an SGD run.

### Splits

AEF's structure is two-way. §S4.1: "Given a training set ... and a validation set
of M embeddings with held out labels, we fit a predictor ... and then report
results on the validation set." Balanced draw plus remainder, and the remainder is
what they report. **There is no third held-out split** — so "match AEF exactly"
and "preserve a test set" are structurally incompatible.

---

## How our protocol differs

### Training set sizes

Every AEF training set is smaller than ours; the ratio ranges from ~2.5× to ~21×.

| dataset | AEF trains on | we train on |
|---|---|---|
| ethiopia | 196 (49×4, balanced) | 574 (imbalanced) |
| africa | 800 (200×4) | ~2,000 |
| descals | 600 (200×3) | 8,049 |
| canada coarse | 612 (68×9) | ~12,800 |
| glance | 3,300 (300×11) | ~28,000 |
| us_trees | 11,700 (300×39) | ~36,000 |

Only the descals and ethiopia figures are measured; the rest assume an 80/10/10
split. Three consequences:

1. **Our absolute numbers are not comparable to AEF's published figures.** We run
   a substantially easier benchmark everywhere except ethiopia.
2. **Our head-to-head comparisons remain fair**, because AEF and Tessera
   embeddings are scored through *our* pipeline on *our* splits.
3. **We are not measuring the regime AEF designed for.** The premise of an
   embedding field is sparse in-situ labels — hundreds of points. Training probes
   on 28k GLanCE labels tests linear decodability given plenty of data, not
   transfer from a handful of field observations.

### Other protocol differences

- **We report the validation split, not test.** `_log_eval_result_to_wandb` is
  called with `val_result` under the `eval/` prefix
  (`evaluator_callback.py:1013`); test goes to a separate prefix. Combined with
  taking the max over 8 learning rates, the reported LP number carries ~8-way
  selection optimism on the split it is reported from. Relative comparisons stay
  fair (baselines use the identical pipeline), but anything external should quote
  test.
- **`select_best_val=False` in our launches**, so the reported value is the *last*
  epoch rather than the best-val epoch (`linear_probe.py:516`). The code logs a
  warning when the final epoch is worse than the best. Flipping this flag is free
  noise reduction.
- **Ethiopia's split is lopsided**: 574 train / 978 val / 977 test — the training
  split is smaller than the eval split.

---

## Implemented: the `balanced_trial` addition

> **Status (2026-08-11): built and on by default for every embedding-eval KNN
> task.** `olmoearth_pretrain/evals/balanced_trial.py` implements the protocol,
> `_aef_ps1_task` attaches a `BalancedTrialConfig` to every `_knn` registration,
> and `_val_embed_probe` runs it on the embeddings the KNN job already holds.
> The defaults reproduce AEF exactly — pool every split, cap per Table 1, draw
> `min(cap, least class)` per class, evaluate on the remainder, repeat for
> `k = 1000 / (2·log₁₀ c′)` draws. The precomputed baselines run these same task
> objects, so `--model=aef` and `--model=tessera_v2_precomputed` inherit it.
> The sections below describe the design as built; the defaults chosen differ
> from the original proposal, which recommended the train-only/fixed-val
> configuration.


**Purely additive. Nothing is removed.** The 8-LR SGD probe on the full imbalanced
split answers a legitimate and different question — given plenty of labels and
real-world class skew, how linearly decodable is this embedding? — and it is what
all our historical numbers measure. The balanced-trial numbers sit beside it.

Running both protocols on the same embeddings separates confounds that are
currently entangled in a single number:

- **full split vs balanced trial** isolates class skew (label budget stays large)
- **a 10-shot draw vs the max-trial draw** isolates label budget at fixed balance
- **ridge vs SGD on the same split** isolates the probe's own contribution

### Where it runs: the kNN job

No new forward pass. `_val_embed_probe` (`evaluator_callback.py:560`) already
materializes train/val (and test, under `run_on_test`) embeddings in memory before
dispatching to either mode, and the kNN job is the right host:

- **It is the only single-instance job.** `embedding_eval_sweep.py:255-270` emits
  8 LP jobs (one per LR) but exactly one kNN job, so the trials compute once
  instead of 8 redundant times.
- **It is the cheapest job** — extraction plus a neighbor lookup — so
  millisecond-scale ridge fits disappear into the extraction time. The LP jobs are
  the 50-epoch long poles.
- **kNN needs train embeddings anyway** as its reference set, so everything
  required is in memory at the call site (after the dequantization block, ~line
  630).

Hook it in `_aef_ps1_task` (`all_evals.py:1413`) gated on mode, so all `_knn`
registration sites inherit it:

```python
balanced_trial=BalancedTrialConfig() if eval_mode == EvalMode.KNN else None,
```

Helpfully, `_aef_ps1_task` sets no `max_train_samples`, so the drawer sees the full
train split. And because the call site is after dequantization, trials run on the
same int8-round-tripped embeddings as kNN and LP.

### Each trial predictor is its own task

A trial result shares nothing with its host task's number except the embeddings:
different training set (a balanced draw, not the train split), different eval set
(the remainder, not val), often a different predictor. Reporting them together
would put two numbers behind one name — and on ethiopia the host's kNN balanced
accuracy is **0.49** while the trial's kNN *at the same k, on the same
embeddings* is **0.77**. Nothing in a metric-name prefix would reliably stop
someone reading the wrong one.

So each predictor is reported as a **separate task**, named
`{host}_aeftrial_{predictor}` with the host's `_knn` suffix dropped (a ridge
result under a `_knn` name reads as a contradiction):

```
eval/ethiopia_..._sentinel2_knn                      <- host kNN, imbalanced train -> val
eval/ethiopia_..._sentinel2_aeftrial_ridge           <- balanced draw -> remainder, ridge
eval/ethiopia_..._sentinel2_aeftrial_knn5            <- balanced draw -> remainder, kNN k=5
eval/ethiopia_..._sentinel2_aeftrial_knn20           <- balanced draw -> remainder, kNN k=20
eval_other/ethiopia_..._sentinel2_aeftrial_ridge/{accuracy,macro_f1,auroc,prauc,...}
```

That reuses the ordinary key layout — `eval/{task}` for the primary metric,
`eval_other/{task}/*` for the rest — so the CSV export and dashboards treat a
trial as just another task. Two things in
`get_max_eval_metrics_from_wandb.py` had to learn about them: the has-test gate
(a trial has no test split of its own) and the task-config lookup (a synthetic
name is not in the registry, so it would `KeyError`).

Each trial task carries, besides the metrics:

| key | meaning |
|---|---|
| `balanced_accuracy` | AEF's reported number: mean over the k draws (the task's primary) |
| `balanced_accuracy_std` | spread *across draws* — how much the draw itself moves the score |
| `balanced_accuracy_sem` | error on the mean, `std/√k` — the error bar AEF's figures carry |
| `n_per_class`, `least_class`, `n_folds`, `pool_size`, `train_size`, `eval_size` | protocol diagnostics, repeated on every predictor so each row is self-describing |

Accuracy, macro F1, AUROC and PR-AUC come out alongside balanced accuracy for
every predictor, because each fold is scored through the shared
`classification_metrics`. The both-metrics-must-agree guardrail from earlier in
this document is therefore automatic here rather than a manual audit.

`std` and `sem` differ by √k, which is a ~17× difference at k = 296 — quote the
wrong one and a contested cell looks decided.

### Draw pool and eval split are independent choices

```python
@dataclass
class BalancedTrialConfig:
    enabled: bool = True
    draw_pool: tuple[str, ...] = ("train", "val", "test")  # AEF pools everything
    eval_split: str = "remainder"             # or a split held out of draw_pool
    n_per_class: int | None = None            # None -> min(cap, least class)
    cap: int = 300                            # 200 for africa/descals, per Table 1
    n_folds: int | None = None                # None -> AEF formula
    max_folds: int | None = None              # None -> keep AEF's count
    seed: int = 0
    knn_ks: tuple[int, ...] = (5, 20)         # 20 matches our headline kNN cell
    ridge_lambda: float = 0.0                 # AEF specifies 0
    fit_intercept: bool = True
    # raises if eval_split is also in draw_pool
```

Three configurations with distinct jobs:

| configuration | draw pool | eval | purpose |
|---|---|---|---|
| **AEF reconciliation (the default)** | all splits | remainder | reproduce published figures; spends the test set |
| **deliberate comparison** | train ∪ val | test (fixed) | clean eval, tightest error bars |
| **cheapest** | train | val (fixed) | identical eval rows to the existing LP/kNN cells |

The default is the first because exact comparability to AEF's published numbers
was the point. The second is arguably a *better* estimator than the protocol it
imitates — fixed eval rows mean the fold spread isolates training-draw variance
instead of mixing in eval-set variance the way a per-fold remainder does — and
`embedding_eval_sweep.py --balanced_trial_draw_pool=train,val
--balanced_trial_eval_split=test` selects it.

**What the default costs.** The remainder contains test rows, so these numbers
are not held-out and training on those points cannot be undone. The headline
train → val kNN/LP cells are untouched — every historical number and every
readiness-page comparison still comes from a protocol that never saw test — but
the trial columns are a spent surface, and repeatedly reporting them across many
configs erodes it further through multiple comparisons.

Label-budget arithmetic for ethiopia, which is why the pool is all three splits:
AEF's 49 per class is half the least class across all 2,530 points. Our
574-window train split is 22.7% of that pool, so a train-only draw would find a
least class near 22 and draw 11 per class; adding val (1,552 total, 61%) draws
~15; only the full pool reaches 48. A train-only trial there is a 10-shot trial
wearing a max-trial label. **`n_per_class` is logged for exactly this reason** —
check it against Table 1 per dataset before quoting a number as
AEF-comparable.

### The module

`olmoearth_pretrain/evals/balanced_trial.py`:

```python
def aef_num_folds(least_class_count) -> int          # S4.3, exact on their table
def draw_balanced_indices(labels, n_per_class, generator) -> Tensor
def fit_ridge_ovr(X, y, num_classes, lam, fit_intercept) -> (W, b)
def run_balanced_trials(...) -> BalancedTrialResult  # mean/std/sem/per-fold
```

Four details that mattered:

- **`RidgeClassifier` semantics** — one-vs-rest, {−1, +1} targets, closed-form
  solve, centering rather than penalizing the intercept, argmax at inference.
  Not softmax, not cross-entropy. Validated against sklearn to 1e-8 at λ = 1 and
  1e-6 against its `svd` solver at λ = 0 (sklearn's default cholesky path is
  singular for an unpenalized fit, so the test has to name the solver).
- **The underdetermined case.** Ethiopia's draw is 49×4 = 196 rows against
  D = 128 (nearly interpolating), and against a d768 arm it is 196 rows against
  768 columns, where λ = 0 is ill-posed. `torch.linalg.pinv` gives the min-norm
  solution and the fit logs a warning; `ridge_lambda > 0` regularizes instead.
- **Reuse `metrics.py`** for per-fold metrics, so BA, accuracy, macro F1, AUROC
  and PR-AUC all come out together — the both-must-agree guardrail is automatic
  rather than a manual audit.
- **Fold seeds are `seed + fold_idx`**, and labels outside `[0, num_classes)`
  are dropped before drawing.

Measured cost on CPU at real scale: 19 ms/fold for an ethiopia-sized pool
(2.5k × 128), ~1 s/fold for a us_trees-sized one (45k rows, 39 classes,
300/class) — so ~3 min for the full 202 folds on the largest task, against the
~75 min of embedding extraction it rides along with. `max_folds` is therefore
left unset by default (AEF's k), not capped at 50 as originally proposed.

### Scope: classification tasks only, not PASTIS

Gate on `task_type == CLASSIFICATION`. PASTIS is excluded for two reasons:

1. **Squared-error one-vs-rest is the wrong objective for mIoU** on a 20-class
   segmentation with a dominant background class. This is the decisive argument.
2. **PASTIS never had the balanced-accuracy pathology.** IoU puts false positives
   in the denominator, so it is precision-aware by construction and cannot be
   inflated by spreading predictions.

Note that *memory* is not the obstacle, contrary to first intuition. Closed-form
ridge never materializes X: XᵀX is D×D (64 KB at D=128) and XᵀY is D×C, both
accumulated in a streaming pass with memory independent of N. Today's SGD path
holds the full embedding tensor only because 50 epochs × 8 LRs need random access
to it (`linear_probe.py:343,448`). For a single-pass closed-form fit, PASTIS-scale
data would be *cheaper* than what we run now — roughly 400 passes reduced to one.
The blocker is the loss, not the size.

### Are the numbers directly comparable to AEF's published figures?

For **ridge balanced accuracy**, yes on protocol — and that is the only cell for
which the claim holds. Four things still stand between an `_aeftrial_ridge`
value and Table 1.

**What now matches.** Balanced draw of `min(cap, least class)` per class, pooled
over every split; remainder as the eval set; `k = 1000/(2·log₁₀ c′)` draws;
`RidgeClassifier` at λ = 0 with argmax; balanced accuracy as the reported metric;
per-dataset caps (300, except 200 for africa and descals) from Table 1.

**What does not, and how to check each one.**

1. **The label pool is not their label pool.** Our year-aligned re-exports
   resolve to the windows where the required imagery exists — lcmap 26,409 of
   26,513 and us_trees 44,886 of 45,382 under Tessera's coverage, the S1+S2+gse
   intersection elsewhere. A different pool means a different least class, hence
   a different `n_per_class` and `k`. **Check `pool_size`, `least_class` and
   `n_per_class` against Table 1 per dataset before quoting anything;** if
   `n_per_class` is 44 where AEF had 49, the trial is not their trial. Check
   `eval_classes == pool_classes` in the same breath — that one is not a
   comparability caveat but a correctness gate.
2. **kNN's `k` is not specified** in the supplement as we have it, so
   `_aeftrial_knn5` / `_aeftrial_knn20` are internally comparable but not
   comparable to their kNN column. Ridge is the specified predictor; prefer it for external claims.
3. **Their published F1 is threshold-handicapped.** §S4.1 tunes the binarization
   threshold to maximize balanced accuracy and computes everything else at that
   threshold, so only their BA is worth comparing against. Our own `macro_f1`
   on the trial tasks, computed through this pipeline, is unaffected.
4. **Imagery and embedding provenance still differ for OlmoEarth.** Our
   embeddings are per-pixel, int8 round-tripped, computed from ws16 windows of
   our own exports. That is a property of the model being scored, not of the
   protocol — but it means "we beat their published number" is a claim about two
   different input pipelines unless the baseline check below passes.

**The confirmation run.** `embedding_eval_sweep.py --model=aef` scores AEF's own
published embedding product through this exact protocol, so its `_aeftrial_ridge`
values should land near their Table 1 figures. That is the test of whether the
implementation reproduces the protocol; run it before quoting any OlmoEarth
number as AEF-comparable. `--model=tessera_v2_precomputed` (and `tessera`,
`tessera_v11`) gives the same treatment to Tessera, which has no published
figures under this protocol — those numbers are ours to establish.

### Not implemented

- **A low-shot arm.** `n_per_class ∈ {1, 10}` reproduces AEF's 1-shot and 10-shot
  variants and costs nothing extra once the embeddings are in memory; it needs a
  second `BalancedTrialConfig` per task and a metric-prefix scheme to keep the
  keys apart.
- **Bootstrap for the k = 1 case** (§S4.4, B = 100 resamples of the eval split),
  because no dataset of ours has an equal-sized pool — and there the balanced
  draw consumes everything, so the trial declines with a warning rather than
  scoring on its own training rows.

If ethiopia's fold std comes back at ±3 points, that retroactively justifies the
noise floor we have been asserting; if it is ±1, several cells we currently treat
as contested become real.

---

## Open questions

- **Are our train splits class-balanced?** Load-bearing and unmeasured — the
  labels are on weka. The eval split is definitely imbalanced (Tessera's kNN
  accuracy 0.8395 vs BA 0.4894 proves it). If train is imbalanced too, we run a
  protocol AEF never ran, which alone could explain part of the ethiopia LP gap.
  The trials now log `least_class` and `pool_size` per dataset, so the
  first kNN job that lands answers this for the pooled set.
- **Does ridge match the SGD probe** on the eight classification tasks? Decides
  whether the balanced-trial numbers are directly comparable to the headline ones.
- **Is Tessera's ethiopia advantage sample efficiency or representation content?**
  A 10-shot arm plus an MLP-capacity probe would separate these. Their embedding
  is 128-d like cand_ndvi's, so it is not a dimensionality effect.
- **Should we add a low-shot arm permanently?** `n_per_class ∈ {10, max}` costs
  nothing extra once embeddings are in memory and measures the sparse-label regime
  our current suite structurally cannot.
