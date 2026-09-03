"""AEF-protocol balanced trials over already-materialized embeddings.

This reproduces the evaluation protocol of *AlphaEarth Foundations* (arXiv
2507.22291v2, Table 1 and supplemental S4) rather than our own probe protocol:

1. **Sampling.** Pool every split's embeddings, then draw a *class-balanced*
   training set of ``n_per_class = min(cap, size of the least-populated class)``
   points. Everything not drawn is the eval set (AEF has no third held-out
   split -- S4.1: "given a training set ... and a validation set of M embeddings
   with held out labels, we fit a predictor ... and then report results on the
   validation set").
2. **k draws.** Repeat the draw ``k = 1000 / (2 * log10(c'))`` times, where
   ``c'`` is the least-present class (S4.3), and report the mean and standard
   deviation across draws. ``k = 1`` when the classes are already equal-sized,
   since only one balanced draw exists.
3. **Predictors.** A ``RidgeClassifier`` with lambda = 0 -- one-vs-rest ordinary
   least squares against {-1, +1} targets, argmax at inference -- plus kNN
   (S4.1). Neither has a model-selection step, which is what makes a few hundred
   refits affordable: each ridge fit is a closed-form solve.

   Optionally, and outside AEF's protocol, the same draws can also be scored
   with extra ridge penalties (``ridge_lambdas``, reported as ``ridge_lam{l}``)
   and with off-the-shelf classifiers at library defaults (``classifiers``:
   random forest, xgboost, logistic regression; see evals/classifier_probes.py).
   Each is its own predictor name, so the AEF-faithful ``ridge``/``knn*`` cells
   are untouched and a forest's number can never be read as the ridge's.

This is purely additive: the caller's own train -> val probe result is untouched.
Each trial predictor is reported as its OWN task -- ``{host}_aeftrial_{predictor}``
with the host's ``_knn`` suffix dropped -- rather than as extra metrics on the
host task. Filing them together would put two numbers behind one name that share
only the embeddings: on ethiopia the host's kNN balanced accuracy is 0.49 while
the trial's kNN at the same k is 0.77, and nothing in a metric name would tell
them apart. See docs/EvalMetricsAndBalancedTrials.md for why the two protocols
answer different questions.

Because a trial draws from *every* split by default, the eval remainder contains
test rows. That is the AEF protocol, but it means these numbers spend the test
set; see the module docstring's caveat in the doc above before treating them as
a held-out result.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch

from olmoearth_pretrain.evals.classifier_probes import (
    ClassifierProbeConfig,
    classifier_scores,
)
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig
from olmoearth_pretrain.evals.knn import _run_knn_for_k
from olmoearth_pretrain.evals.metrics import (
    EvalMetric,
    EvalResult,
    classification_metrics,
)

logger = logging.getLogger(__name__)

# Marks the synthetic task names the trials report under, e.g.
# "ethiopia_crops_..._sentinel2_aeftrial_ridge". The protocol lives in the TASK
# name rather than in a metric prefix, so a trial number can never be read as
# the host task's own kNN number -- they differ by ~25 points on ethiopia and
# are otherwise indistinguishable (same embeddings, same k, same metric name).
TASK_SUFFIX = "aeftrial"


def trial_task_name(host_task: str, predictor: str) -> str:
    """Synthetic task name a trial predictor's result is reported under.

    The host is the KNN twin the trials ride on; its ``_knn`` suffix is dropped
    because a ridge result under a ``_knn`` task name reads as a contradiction.
    """
    return f"{host_task.removesuffix('_knn')}_{TASK_SUFFIX}_{predictor}"


def lambda_tag(lam: float) -> str:
    """Predictor-name-safe rendering of a ridge penalty: 1.0 -> "lam1", 0.1 -> "lam0p1"."""
    return "lam" + f"{lam:g}".replace(".", "p").replace("-", "m").replace("+", "")


# Metrics aggregated (mean + std) across folds. Balanced accuracy is AEF's
# reported metric; macro F1 and accuracy are the disagreement detectors from
# docs/EvalMetricsAndBalancedTrials.md, and the two ranking metrics are
# threshold-free third opinions.
AGGREGATED_METRICS: tuple[str, ...] = (
    EvalMetric.BALANCED_ACCURACY.value,
    EvalMetric.ACCURACY.value,
    EvalMetric.MACRO_F1.value,
    EvalMetric.AUROC.value,
    EvalMetric.PRAUC.value,
)

# Draw from every split and evaluate on the remainder: AEF's two-way structure.
ALL_SPLITS: tuple[str, ...] = ("train", "val", "test")
REMAINDER = "remainder"


@dataclass
class BalancedTrialConfig:
    """Configuration for the AEF balanced-trial protocol.

    The defaults reproduce AEF exactly: draw from the union of all splits, cap
    the per-class draw at 300, evaluate on the remainder, and use the S4.3 fold
    formula.
    """

    # Set False to skip the trials without having to null out the whole config
    # (nested optionals are awkward to override from the sweep CLI).
    enabled: bool = True
    # Splits pooled to draw the balanced training set from. AEF pools
    # everything; ("train",) or ("train", "val") keep a split unspent at the
    # cost of a smaller least-class count (and so a smaller balanced draw).
    draw_pool: tuple[str, ...] = ALL_SPLITS
    # "remainder" (AEF: everything in the pool that was not drawn, so the eval
    # rows change per fold) or a fixed split name not in ``draw_pool``, which
    # holds the eval rows constant so the fold spread isolates training-draw
    # variance.
    eval_split: str = REMAINDER
    # None -> min(cap, least_class * least_class_draw_fraction).
    n_per_class: int | None = None
    # AEF's deliberate cap, "meant to represent more realistic sparse dataset
    # sizes (hundreds as opposed to thousands or millions of points)". 200 and
    # 150 are used for some of their tasks; 300 is the common value.
    cap: int = 300
    # SAFETY NET, not the mechanism: the largest share of the LEAST-populated
    # class the draw may take. Drawing the whole of it removes that class from
    # the remainder entirely, and balanced accuracy then silently averages over
    # K-1 classes -- dropping the rarest and usually hardest one, so the score
    # goes up. `eval_classes` vs `pool_classes` on every result is the check.
    #
    # What normally sets the draw is ``cap``, taken per dataset from AEF's
    # Table 1 (see AEF_MAX_TRIAL_CAPS). Every one of our least classes exceeds
    # its dataset's cap, so the cap binds first everywhere and this fraction
    # never fires -- it exists for a future dataset whose rarest class falls
    # below its cap.
    #
    # 0.9 rather than something smaller because the fraction is now pure
    # insurance: lowering it would only shrink the draw below AEF's budget on
    # datasets where the cap already protects the eval set. An earlier version
    # used 0.5 on the theory that AEF's odd Table 1 values were half the least
    # class; measured counts across all eight datasets contradicted that, and
    # 0.5 forfeited budget parity everywhere the cap binds.
    least_class_draw_fraction: float = 0.9
    # None -> the S4.3 formula (or 1 when the pool is already class-balanced).
    n_folds: int | None = None
    # Backstop on the formula's 200-500 folds. None keeps AEF's count; the
    # standard error only falls as 1/sqrt(k), so a cap is cheap insurance if
    # the trials ever become the long pole.
    max_folds: int | None = None
    seed: int = 0
    # kNN neighbor counts to run alongside ridge. 20 matches our headline kNN
    # cell; 5 is a more reasonable choice against a few-hundred-point balanced
    # reference set.
    knn_ks: tuple[int, ...] = (5, 20)
    # AEF specifies lambda = 0 (ordinary least squares). A small positive value
    # regularizes the underdetermined case (draw rows < embedding dims) instead
    # of relying on the min-norm pseudoinverse solution.
    ridge_lambda: float = 0.0
    fit_intercept: bool = True
    # Extra ridge penalties fit on the same draws, each reported as its own
    # predictor ``ridge_lam{lambda}`` (1.0 -> "ridge_lam1", 0.1 -> "ridge_lam0p1").
    # sklearn's RidgeClassifier default is alpha=1; AEF's lambda=0 is a deliberate
    # no-hyperparameter choice whose price is an unstable estimator when the draw
    # is small relative to the embedding width (ethiopia at d128: 196 rows).
    ridge_lambdas: tuple[float, ...] = ()
    # Off-the-shelf classifiers fit per fold on the same draws; ``names`` empty
    # (the default) runs none. Reported under their own predictor names
    # ("rf", "xgb", "logreg").
    classifiers: ClassifierProbeConfig = field(default_factory=ClassifierProbeConfig)


@dataclass
class BalancedTrialResult:
    """One EvalResult per predictor, plus the per-fold values they came from."""

    # Predictor name ("ridge", "knn5", ...) -> its aggregated EvalResult. Each
    # is reported as its own task by the caller, because a trial result shares
    # nothing with the host task's number except the embeddings: different
    # training set, different eval set, often a different predictor.
    results: dict[str, EvalResult] = field(default_factory=dict)
    # predictor name -> metric name -> one value per fold.
    per_fold: dict[str, dict[str, list[float]]] = field(default_factory=dict)


def aef_num_folds(trial_size: int) -> int:
    """AEF's S4.3 fold count ``k = 1000 / (2 * log10(c'))``.

    ``c'`` is **the per-class trial size**, not the least-populated class.
    S4.3's prose calls it "the least-present class", but their published fold
    counts are reproduced only by their Table 1 max-trial column: 49 -> 296
    (ethiopia), 68 -> 273 (canada coarse), 300 -> 202 (the capped datasets).
    Those two readings coincide whenever the trial size is set by the least
    class, which is why the wording is ambiguous; they diverge once a cap or a
    draw fraction binds, and then the trial size is what matches their numbers.

    Folds go *up* as the trial gets smaller, and only logarithmically, so the
    formula never leaves the 200-500 band for any plausible ``c'``.
    """
    if trial_size <= 1:
        # log10(1) = 0; a single-point-per-class trial admits no sampling
        # variance worth hundreds of draws.
        return 1
    return max(1, int(round(1000.0 / (2.0 * math.log10(trial_size)))))


def draw_balanced_indices(
    labels: torch.Tensor, n_per_class: int, generator: torch.Generator
) -> torch.Tensor:
    """Draw ``n_per_class`` indices without replacement from each present class.

    Args:
        labels: Integer class labels of shape (N,), already filtered to valid
            classes.
        n_per_class: Number to draw per class. Classes with fewer members
            contribute all of theirs.
        generator: Seeded CPU generator; the draw is a function of it alone.

    Returns:
        Sorted indices into ``labels`` of the drawn rows.
    """
    parts = []
    for class_value in torch.unique(labels).tolist():
        class_indices = (labels == class_value).nonzero(as_tuple=True)[0]
        permutation = torch.randperm(class_indices.numel(), generator=generator)
        parts.append(class_indices[permutation[:n_per_class]])
    return torch.cat(parts).sort().values


def fit_ridge_ovr(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    lam: float = 0.0,
    fit_intercept: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit scikit-learn's ``RidgeClassifier`` in closed form.

    Matches its semantics: one-vs-rest against {-1, +1} targets, centering
    rather than penalizing the intercept, and a plain least-squares solve. The
    caller argmaxes the resulting decision values, which is what
    ``RidgeClassifier.predict`` does for multiclass.

    ``lam = 0`` (AEF's choice) makes the normal equations singular whenever the
    draw has fewer rows than embedding dimensions -- ethiopia's 49x4 = 196 rows
    against a 768-d arm, say -- so the fit falls back to the pseudoinverse's
    minimum-norm solution and logs a warning.

    Args:
        embeddings: Training embeddings of shape (N, D).
        labels: Integer class labels of shape (N,).
        num_classes: Total number of classes (one decision column each).
        lam: Ridge penalty. 0 = ordinary least squares.
        fit_intercept: Whether to fit an unpenalized intercept.

    Returns:
        ``(weights, bias)`` of shapes (D, num_classes) and (num_classes,), in
        float64.
    """
    features = embeddings.double()
    n_samples, n_features = features.shape
    targets = torch.full(
        (n_samples, num_classes), -1.0, dtype=torch.float64, device=features.device
    )
    targets[torch.arange(n_samples, device=features.device), labels.long()] = 1.0

    if fit_intercept:
        feature_mean = features.mean(dim=0, keepdim=True)
        target_mean = targets.mean(dim=0, keepdim=True)
    else:
        feature_mean = torch.zeros(
            (1, n_features), dtype=torch.float64, device=features.device
        )
        target_mean = torch.zeros(
            (1, num_classes), dtype=torch.float64, device=features.device
        )
    centered_features = features - feature_mean
    centered_targets = targets - target_mean

    if lam > 0:
        gram = centered_features.T @ centered_features
        gram += lam * torch.eye(n_features, dtype=torch.float64, device=features.device)
        weights = torch.linalg.solve(gram, centered_features.T @ centered_targets)
    else:
        if n_samples <= n_features:
            logger.warning(
                f"Ridge fit is underdetermined ({n_samples} rows, {n_features} dims) "
                f"with lambda=0; using the minimum-norm pseudoinverse solution. Set "
                f"ridge_lambda > 0 to regularize instead."
            )
        weights = torch.linalg.pinv(centered_features) @ centered_targets

    bias = (target_mean - feature_mean @ weights).squeeze(0)
    return weights, bias


def _resolve_pool(
    embeddings_by_split: dict[str, torch.Tensor | None],
    labels_by_split: dict[str, torch.Tensor | None],
    splits: tuple[str, ...],
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate the requested splits, dropping rows with out-of-range labels."""
    embedding_parts, label_parts = [], []
    missing = []
    for split in splits:
        split_embeddings = embeddings_by_split.get(split)
        split_labels = labels_by_split.get(split)
        if split_embeddings is None or split_labels is None:
            missing.append(split)
            continue
        embedding_parts.append(split_embeddings.detach().cpu().float())
        label_parts.append(split_labels.detach().cpu().reshape(-1).long())
    if missing:
        logger.warning(
            f"Balanced trial: split(s) {missing} are not materialized and will be "
            f"left out of the draw pool (run_on_test=True is required to pool the "
            f"test split). This is no longer the AEF protocol -- the balanced draw "
            f"is made from a smaller pool, so its least-class count is smaller too."
        )
    if not embedding_parts:
        raise ValueError(f"Balanced trial: none of the splits {splits} are available")

    pool_embeddings = torch.cat(embedding_parts, dim=0)
    pool_labels = torch.cat(label_parts, dim=0)
    valid = (pool_labels >= 0) & (pool_labels < num_classes)
    if not bool(valid.all()):
        logger.info(
            f"Balanced trial: dropping {int((~valid).sum())} of {valid.numel()} pooled "
            f"rows with labels outside [0, {num_classes})"
        )
        pool_embeddings = pool_embeddings[valid]
        pool_labels = pool_labels[valid]
    return pool_embeddings, pool_labels


def _mean_std_sem(values: list[float]) -> tuple[float, float, int]:
    """Mean, sample standard deviation, and contributing count.

    NaNs are ignored: AUROC/PR-AUC are undefined for a class with no positives
    or no negatives in a fold's eval set, and a fold that produced one should
    not poison the average.
    """
    finite = [v for v in values if not math.isnan(v)]
    if not finite:
        return float("nan"), float("nan"), 0
    mean = sum(finite) / len(finite)
    if len(finite) < 2:
        return mean, 0.0, len(finite)
    variance = sum((v - mean) ** 2 for v in finite) / (len(finite) - 1)
    return mean, math.sqrt(variance), len(finite)


def run_balanced_trials(
    config: EvalDatasetConfig,
    embeddings_by_split: dict[str, torch.Tensor | None],
    labels_by_split: dict[str, torch.Tensor | None],
    trial_config: BalancedTrialConfig,
    device: torch.device,
) -> BalancedTrialResult:
    """Run the AEF balanced-trial protocol on materialized embeddings.

    Args:
        config: Dataset config (supplies ``num_classes``).
        embeddings_by_split: "train"/"val"/"test" -> (N, D) embeddings. Missing
            or None splits are skipped.
        labels_by_split: The matching integer labels.
        trial_config: Protocol settings.
        device: Device the kNN distance computations run on.

    Returns:
        A :class:`BalancedTrialResult` whose ``metrics`` are flat ``bt_*`` keys
        ready to log alongside the headline eval metrics.

    Raises:
        ValueError: If ``eval_split`` is a split that is also in ``draw_pool``
            (which would leak training rows into the eval set).
    """
    if not trial_config.enabled:
        return BalancedTrialResult()

    num_classes = config.num_classes
    if trial_config.eval_split != REMAINDER:
        if trial_config.eval_split in trial_config.draw_pool:
            raise ValueError(
                f"Balanced trial: eval_split '{trial_config.eval_split}' is also in "
                f"draw_pool {trial_config.draw_pool}, which would train on eval rows"
            )

    pool_embeddings, pool_labels = _resolve_pool(
        embeddings_by_split, labels_by_split, trial_config.draw_pool, num_classes
    )

    present_classes, class_counts = torch.unique(pool_labels, return_counts=True)
    if present_classes.numel() < 2:
        logger.warning(
            f"Balanced trial: pool has {present_classes.numel()} class(es); skipping"
        )
        return BalancedTrialResult()
    least_class_count = int(class_counts.min())

    # Leave part of the least-populated class in the remainder: a class drawn in
    # full disappears from the eval set, and balanced accuracy then averages
    # over the classes that remain without saying so.
    from_least_class = max(
        1, int(least_class_count * trial_config.least_class_draw_fraction)
    )
    n_per_class = trial_config.n_per_class or min(trial_config.cap, from_least_class)
    if n_per_class < 1:
        logger.warning("Balanced trial: n_per_class resolved to 0; skipping")
        return BalancedTrialResult()
    if n_per_class > least_class_count:
        logger.warning(
            f"Balanced trial: n_per_class={n_per_class} exceeds the least-populated "
            f"class ({least_class_count}); that class contributes all of its rows and "
            f"the draw is not fully balanced"
        )

    # Which classes the draw would consume entirely. In remainder mode they are
    # absent from the eval set, so every metric is computed over fewer classes
    # than the task has -- silently, and in the direction that flatters the
    # score, since the starved classes are the rarest and usually the hardest.
    starved = [
        (int(c), int(n))
        for c, n in zip(present_classes.tolist(), class_counts.tolist())
        if n <= n_per_class
    ]
    if starved and trial_config.eval_split == REMAINDER:
        logger.warning(
            f"Balanced trial: {len(starved)} of {present_classes.numel()} classes are "
            f"consumed entirely by a {n_per_class}/class draw {starved} and will be "
            f"ABSENT from the eval remainder, so metrics average over "
            f"{present_classes.numel() - len(starved)} classes instead of "
            f"{present_classes.numel()}. Lower least_class_draw_fraction or cap."
        )

    draw_size = int(class_counts.clamp(max=n_per_class).sum())
    if trial_config.eval_split == REMAINDER and draw_size >= pool_labels.numel():
        logger.warning(
            f"Balanced trial: a {n_per_class}/class draw consumes the entire "
            f"{pool_labels.numel()}-row pool, leaving no remainder to evaluate on. "
            f"Lower cap/n_per_class, or set eval_split to a split held out of "
            f"draw_pool. Skipping."
        )
        return BalancedTrialResult()

    if trial_config.n_folds is not None:
        n_folds = trial_config.n_folds
    elif int(class_counts.max()) == n_per_class:
        # Every class is already exactly the draw size, so there is only one
        # balanced draw and no sampling variance to average over (S4.3).
        n_folds = 1
    else:
        n_folds = aef_num_folds(n_per_class)
    if trial_config.max_folds is not None and n_folds > trial_config.max_folds:
        logger.info(
            f"Balanced trial: capping folds at max_folds={trial_config.max_folds} "
            f"(AEF's formula gives {n_folds} for c'={n_per_class})"
        )
        n_folds = trial_config.max_folds

    # Fixed-eval mode: eval rows come from a split held out of the pool, so the
    # fold spread isolates training-draw variance instead of also moving the
    # eval set. Remainder mode is AEF's own two-way structure.
    fixed_eval: tuple[torch.Tensor, torch.Tensor] | None = None
    if trial_config.eval_split != REMAINDER:
        fixed_eval = _resolve_pool(
            embeddings_by_split,
            labels_by_split,
            (trial_config.eval_split,),
            num_classes,
        )

    logger.info(
        f"Balanced trial: pool={pool_embeddings.shape[0]} rows over "
        f"{present_classes.numel()}/{num_classes} classes (least class "
        f"{least_class_count}), drawing {n_per_class}/class x {n_folds} fold(s), "
        f"eval={'remainder' if fixed_eval is None else trial_config.eval_split}"
    )

    pool_embeddings_device = pool_embeddings.to(device)
    if fixed_eval is not None:
        fixed_eval_embeddings = fixed_eval[0].to(device)
        fixed_eval_labels = fixed_eval[1]

    trial_config.classifiers.validate()
    extra_ridge = [
        (lam, f"ridge_{lambda_tag(lam)}") for lam in trial_config.ridge_lambdas
    ]
    classifier_names = list(trial_config.classifiers.names)
    predictors = (
        ["ridge"]
        + [name for _, name in extra_ridge]
        + [f"knn{k}" for k in trial_config.knn_ks]
        + classifier_names
    )
    if classifier_names:
        # The estimators are CPU numpy; index the pool once rather than moving
        # a fold's rows off the device every time.
        pool_embeddings_np = pool_embeddings.detach().cpu().float().numpy()
        pool_labels_np = pool_labels.detach().cpu().long().numpy()
        if fixed_eval is not None:
            fixed_eval_np = fixed_eval[0].detach().cpu().float().numpy()
    per_fold: dict[str, dict[str, list[float]]] = {
        name: {metric: [] for metric in AGGREGATED_METRICS} for name in predictors
    }
    train_sizes: list[int] = []
    eval_sizes: list[int] = []
    eval_class_counts: list[int] = []

    for fold_idx in range(n_folds):
        generator = torch.Generator().manual_seed(trial_config.seed + fold_idx)
        train_indices = draw_balanced_indices(pool_labels, n_per_class, generator)
        train_embeddings = pool_embeddings_device[train_indices]
        train_labels = pool_labels[train_indices]

        if fixed_eval is None:
            held_out = torch.ones(pool_labels.numel(), dtype=torch.bool)
            held_out[train_indices] = False
            eval_embeddings = pool_embeddings_device[held_out]
            eval_labels = pool_labels[held_out]
        else:
            eval_embeddings = fixed_eval_embeddings
            eval_labels = fixed_eval_labels
        if eval_labels.numel() == 0:
            logger.warning("Balanced trial: eval set is empty; skipping")
            return BalancedTrialResult()
        train_sizes.append(int(train_labels.numel()))
        eval_sizes.append(int(eval_labels.numel()))
        eval_class_counts.append(int(torch.unique(eval_labels).numel()))

        weights, bias = fit_ridge_ovr(
            train_embeddings,
            train_labels.to(train_embeddings.device),
            num_classes,
            lam=trial_config.ridge_lambda,
            fit_intercept=trial_config.fit_intercept,
        )
        # Decision values, not probabilities: argmax reproduces
        # RidgeClassifier.predict and the ranking metrics only need an order.
        ridge_scores = (eval_embeddings.double() @ weights + bias).float().cpu()
        _record_fold(
            per_fold["ridge"],
            predictions=ridge_scores.argmax(dim=1),
            labels=eval_labels,
            scores=ridge_scores,
        )

        for lam, name in extra_ridge:
            weights, bias = fit_ridge_ovr(
                train_embeddings,
                train_labels.to(train_embeddings.device),
                num_classes,
                lam=lam,
                fit_intercept=trial_config.fit_intercept,
            )
            scores = (eval_embeddings.double() @ weights + bias).float().cpu()
            _record_fold(
                per_fold[name],
                predictions=scores.argmax(dim=1),
                labels=eval_labels,
                scores=scores,
            )

        for k in trial_config.knn_ks:
            knn_scores = _run_knn_for_k(
                train_embeddings=train_embeddings,
                train_labels=train_labels.to(device),
                test_embeddings=eval_embeddings,
                num_classes=num_classes,
                k=min(k, int(train_labels.numel())),
                device=device,
                skip_idx=False,
                return_scores=True,
            )
            _record_fold(
                per_fold[f"knn{k}"],
                predictions=knn_scores.argmax(dim=1),
                labels=eval_labels,
                scores=knn_scores,
            )

        for name in classifier_names:
            train_indices_np = train_indices.numpy()
            eval_features_np = (
                fixed_eval_np
                if fixed_eval is not None
                else pool_embeddings_np[held_out.numpy()]
            )
            clf_scores = classifier_scores(
                name,
                trial_config.classifiers,
                pool_embeddings_np[train_indices_np],
                pool_labels_np[train_indices_np],
                eval_features_np,
                num_classes,
            )
            _record_fold(
                per_fold[name],
                predictions=clf_scores.argmax(dim=1),
                labels=eval_labels,
                scores=clf_scores,
            )

        if n_folds > 1 and (fold_idx + 1) % 25 == 0:
            logger.info(f"Balanced trial: completed {fold_idx + 1}/{n_folds} folds")

    # Protocol diagnostics, repeated on every predictor's result so each row is
    # self-describing: reading a trial number without n_per_class next to it is
    # how a 10-shot trial gets mistaken for a max-trial one.
    diagnostics = {
        "n_per_class": float(n_per_class),
        "n_folds": float(n_folds),
        "least_class": float(least_class_count),
        "pool_size": float(pool_embeddings.shape[0]),
        "train_size": float(train_sizes[0]),
        "eval_size": float(eval_sizes[0]),
        "embedding_dim": float(pool_embeddings.shape[1]),
        # How many classes the metrics were actually averaged over, and how many
        # the pool has. If these differ, the draw starved a class out of the
        # eval set and every score above is over a smaller class set than the
        # task's -- compare them before quoting a number.
        "eval_classes": float(min(eval_class_counts)),
        "pool_classes": float(present_classes.numel()),
    }

    results: dict[str, EvalResult] = {}
    for predictor, fold_metrics in per_fold.items():
        metrics = dict(diagnostics)
        for metric_name, values in fold_metrics.items():
            mean, std, count = _mean_std_sem(values)
            metrics[metric_name] = mean
            # Spread across draws (how much the balanced draw itself moves the
            # score) and the error on the mean of those draws, which is the
            # error bar AEF's figures carry. They differ by sqrt(k), and k is
            # large here, so quoting the wrong one is a ~17x mistake at k=296.
            metrics[f"{metric_name}_std"] = std
            metrics[f"{metric_name}_sem"] = (
                std / math.sqrt(count) if count > 0 and not math.isnan(std) else std
            )
        primary_key = EvalMetric.BALANCED_ACCURACY.value
        results[predictor] = EvalResult(
            primary=metrics[primary_key],
            primary_metric=EvalMetric.BALANCED_ACCURACY,
            primary_metric_key=primary_key,
            metrics=metrics,
        )
        logger.info(
            f"Balanced trial [{predictor}]: balanced accuracy "
            f"{metrics[primary_key]:.4f} (std {metrics[f'{primary_key}_std']:.4f} "
            f"across draws, sem {metrics[f'{primary_key}_sem']:.4f}) over "
            f"{n_folds} fold(s)"
        )

    return BalancedTrialResult(results=results, per_fold=per_fold)


def _record_fold(
    fold_metrics: dict[str, list[float]],
    predictions: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Score one fold with the shared metric code and append to the running lists."""
    result = classification_metrics(
        predictions=predictions,
        labels=labels,
        scores=scores,
        is_multilabel=False,
    )
    for metric_name in fold_metrics:
        fold_metrics[metric_name].append(float(result.metrics[metric_name]))
