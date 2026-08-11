"""Tests for the AEF balanced-trial protocol."""

import math

import pytest
import torch
from sklearn.linear_model import RidgeClassifier

from olmoearth_pretrain.evals.balanced_trial import (
    BalancedTrialConfig,
    aef_num_folds,
    draw_balanced_indices,
    fit_ridge_ovr,
    run_balanced_trials,
    trial_task_name,
)
from olmoearth_pretrain.evals.datasets.configs import EvalDatasetConfig
from olmoearth_pretrain.evals.task_types import TaskType

NUM_CLASSES = 4
EMBEDDING_DIM = 16


def _config(num_classes: int = NUM_CLASSES) -> EvalDatasetConfig:
    return EvalDatasetConfig(
        task_type=TaskType.CLASSIFICATION,
        imputes=[],
        num_classes=num_classes,
        is_multilabel=False,
        supported_modalities=[],
    )


def _imbalanced_labels() -> torch.Tensor:
    """Labels with per-class counts 100 / 60 / 25 / 12 (least class = 12)."""
    counts = [100, 60, 25, 12]
    return torch.cat(
        [
            torch.full((count,), idx, dtype=torch.long)
            for idx, count in enumerate(counts)
        ]
    )


def _separable_embeddings(labels: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """Class-conditional Gaussians around well-separated random means."""
    generator = torch.Generator().manual_seed(seed)
    means = torch.randn(NUM_CLASSES, EMBEDDING_DIM, generator=generator) * 4.0
    noise = torch.randn(labels.numel(), EMBEDDING_DIM, generator=generator)
    return means[labels] + noise


class TestAefNumFolds:
    """The S4.3 fold formula k = 1000 / (2 * log10(c'))."""

    @pytest.mark.parametrize(
        "trial_size,expected",
        [
            # AEF's Table 1 max-trial column against their published fold
            # counts. Keying on the trial size (not the least class) is what
            # reproduces these: folds go up as the trial gets smaller.
            (10, 500),
            (49, 296),  # ethiopia_crops
            (68, 273),  # canada_crops_coarse
            (100, 250),
            (200, 217),  # africa_crop_mask, descals -- capped datasets
            (300, 202),  # lcmap, glance, us_trees -- capped datasets
        ],
    )
    def test_matches_published_values(self, trial_size: int, expected: int) -> None:
        """The formula reproduces AEF's published fold counts exactly."""
        assert aef_num_folds(trial_size) == expected

    def test_singleton_trial_does_not_divide_by_zero(self) -> None:
        """log10(1) == 0; one point per class admits no sampling variance."""
        assert aef_num_folds(1) == 1
        assert aef_num_folds(0) == 1

    def test_stays_in_the_200_to_500_band_over_aefs_own_range(self) -> None:
        """Over Table 1's range it is an engineered "do a couple hundred draws"."""
        # Monotonically decreasing in c', and never outside 200-500 there.
        counts = list(range(10, 301))
        folds = [aef_num_folds(count) for count in counts]
        assert all(200 <= k <= 500 for k in folds)
        assert folds == sorted(folds, reverse=True)


class TestDrawBalancedIndices:
    """The balanced draw itself."""

    def test_exact_per_class_counts(self) -> None:
        """Every class contributes exactly n_per_class rows, without replacement."""
        labels = _imbalanced_labels()
        generator = torch.Generator().manual_seed(0)
        indices = draw_balanced_indices(labels, n_per_class=12, generator=generator)

        drawn = labels[indices]
        _, counts = torch.unique(drawn, return_counts=True)
        assert counts.tolist() == [12] * NUM_CLASSES
        assert indices.numel() == 48
        # Without replacement.
        assert torch.unique(indices).numel() == indices.numel()

    def test_smaller_classes_contribute_everything_they_have(self) -> None:
        """A class smaller than the draw contributes all of its rows."""
        labels = _imbalanced_labels()
        generator = torch.Generator().manual_seed(0)
        indices = draw_balanced_indices(labels, n_per_class=30, generator=generator)

        _, counts = torch.unique(labels[indices], return_counts=True)
        assert counts.tolist() == [30, 30, 25, 12]

    def test_deterministic_in_the_generator_seed(self) -> None:
        """The draw is a function of the seeded generator and nothing else."""
        labels = _imbalanced_labels()
        first = draw_balanced_indices(labels, 10, torch.Generator().manual_seed(7))
        same = draw_balanced_indices(labels, 10, torch.Generator().manual_seed(7))
        different = draw_balanced_indices(labels, 10, torch.Generator().manual_seed(8))

        assert torch.equal(first, same)
        assert not torch.equal(first, different)

    def test_remainder_is_disjoint_from_the_draw(self) -> None:
        """The eval remainder shares no row with the training draw."""
        labels = _imbalanced_labels()
        indices = draw_balanced_indices(labels, 12, torch.Generator().manual_seed(0))

        held_out = torch.ones(labels.numel(), dtype=torch.bool)
        held_out[indices] = False
        remainder = held_out.nonzero(as_tuple=True)[0]

        assert set(indices.tolist()).isdisjoint(set(remainder.tolist()))
        assert indices.numel() + remainder.numel() == labels.numel()


class TestFitRidgeOvr:
    """Closed-form ridge against scikit-learn's RidgeClassifier."""

    @pytest.mark.parametrize("fit_intercept", [True, False])
    def test_matches_sklearn_with_penalty(self, fit_intercept: bool) -> None:
        """Decision values match RidgeClassifier's, intercept or not."""
        labels = _imbalanced_labels()
        embeddings = _separable_embeddings(labels)

        weights, bias = fit_ridge_ovr(
            embeddings,
            labels,
            num_classes=NUM_CLASSES,
            lam=1.0,
            fit_intercept=fit_intercept,
        )
        ours = (embeddings.double() @ weights + bias).numpy()

        sklearn_model = RidgeClassifier(alpha=1.0, fit_intercept=fit_intercept)
        # float64 on sklearn's side too, so the comparison measures our solve
        # rather than its float32 accumulation.
        features = embeddings.double().numpy()
        sklearn_model.fit(features, labels.numpy())
        theirs = sklearn_model.decision_function(features)

        assert ours == pytest.approx(theirs, abs=1e-8)

    def test_matches_sklearn_at_lambda_zero(self) -> None:
        """AEF's unpenalized fit matches sklearn's SVD solver.

        sklearn needs the solver named explicitly there: its default cholesky
        path is singular for an unpenalized fit.
        """
        labels = _imbalanced_labels()
        embeddings = _separable_embeddings(labels)

        weights, bias = fit_ridge_ovr(
            embeddings, labels, num_classes=NUM_CLASSES, lam=0.0
        )
        ours = (embeddings.double() @ weights + bias).numpy()

        sklearn_model = RidgeClassifier(alpha=0.0, solver="svd")
        # float64 on sklearn's side too, so the comparison measures our solve
        # rather than its float32 accumulation.
        features = embeddings.double().numpy()
        sklearn_model.fit(features, labels.numpy())
        theirs = sklearn_model.decision_function(features)

        assert ours == pytest.approx(theirs, abs=1e-6)
        # And the hard predictions -- argmax of the decision values -- agree.
        assert (
            torch.from_numpy(theirs).argmax(dim=1) == torch.tensor(ours).argmax(dim=1)
        ).all()

    def test_underdetermined_draw_uses_the_min_norm_solution(self) -> None:
        """Fewer rows than dimensions still yields a finite, interpolating fit.

        Ethiopia's 49 x 4 = 196-row draw against a 768-d arm is the real case,
        where lambda = 0 is ill-posed.
        """
        labels = torch.arange(NUM_CLASSES).repeat(3)
        embeddings = torch.randn(
            labels.numel(), 64, generator=torch.Generator().manual_seed(0)
        )

        weights, bias = fit_ridge_ovr(
            embeddings, labels, num_classes=NUM_CLASSES, lam=0.0
        )
        scores = embeddings.double() @ weights + bias

        assert torch.isfinite(weights).all()
        # An underdetermined least-squares fit interpolates its targets exactly.
        targets = torch.full((labels.numel(), NUM_CLASSES), -1.0, dtype=torch.float64)
        targets[torch.arange(labels.numel()), labels] = 1.0
        assert scores == pytest.approx(targets.numpy(), abs=1e-6)


class TestRunBalancedTrials:
    """End-to-end protocol behaviour."""

    def _splits(self, seed: int = 0) -> tuple[dict, dict]:
        """Three splits whose pooled per-class counts are 100 / 60 / 25 / 12."""
        labels = _imbalanced_labels()
        embeddings = _separable_embeddings(labels, seed=seed)
        # Interleave so every split carries every class.
        train = torch.arange(0, labels.numel(), 3)
        val = torch.arange(1, labels.numel(), 3)
        test = torch.arange(2, labels.numel(), 3)
        return (
            {
                "train": embeddings[train],
                "val": embeddings[val],
                "test": embeddings[test],
            },
            {"train": labels[train], "val": labels[val], "test": labels[test]},
        )

    def test_pools_every_split_and_sizes_off_the_least_class(self) -> None:
        """The draw pools all three splits and sizes itself off the least class."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(n_folds=3, knn_ks=(5,)),
            device=torch.device("cpu"),
        )

        metrics = result.results["ridge"].metrics
        assert metrics["pool_size"] == 197.0  # 100 + 60 + 25 + 12
        assert metrics["least_class"] == 12.0
        # min(cap=300, 0.9 * least class) -- the cap does not bind here, so
        # the safety-net fraction sets the draw and part of the least class
        # stays in the remainder rather than being consumed by the draw.
        assert metrics["n_per_class"] == 10.0
        assert metrics["train_size"] == 40.0
        assert metrics["eval_size"] == 197.0 - 40.0
        assert metrics["eval_classes"] == metrics["pool_classes"] == 4.0
        assert metrics["n_folds"] == 3.0

    def test_cap_binds_when_the_least_class_is_large(self) -> None:
        """The cap, not the least class, sets the draw when it is the smaller."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(cap=5, n_folds=2, knn_ks=()),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["n_per_class"] == 5.0
        assert result.results["ridge"].metrics["train_size"] == 20.0

    def test_reports_every_predictor_and_metric(self) -> None:
        """Ridge and every requested kNN report the full metric set per fold."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(n_folds=4, knn_ks=(5, 20)),
            device=torch.device("cpu"),
        )

        for predictor in ("ridge", "knn5", "knn20"):
            for metric in ("balanced_accuracy", "accuracy", "macro_f1"):
                assert metric in result.results[predictor].metrics
                assert f"{metric}_std" in result.results[predictor].metrics
            # Separable classes, so a sane protocol scores well above chance.
            assert result.results[predictor].metrics["balanced_accuracy"] > 0.8
            assert len(result.per_fold[predictor]["balanced_accuracy"]) == 4

    def test_fold_count_defaults_to_the_aef_formula(self) -> None:
        """With no explicit n_folds the S4.3 formula applies, then max_folds caps it."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(max_folds=5, knn_ks=()),
            device=torch.device("cpu"),
        )
        # The formula keys on the TRIAL SIZE, which is 0.9 * the least class
        # of 12: 1000 / (2 * log10(10)) = 500, then capped to 5.
        assert aef_num_folds(10) == 500
        assert result.results["ridge"].metrics["n_per_class"] == 10.0
        assert result.results["ridge"].metrics["n_folds"] == 5.0

    def test_single_fold_when_the_pool_is_already_balanced(self) -> None:
        """S4.3: k = 1 when the classes are already equal-sized.

        Only one balanced draw exists then, so there is no sampling variance to
        average over. That draw consumes the whole pool, so it needs an eval
        split held out of it.
        """
        labels = torch.arange(NUM_CLASSES).repeat(20)
        embeddings = _separable_embeddings(labels)
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": embeddings, "test": embeddings},
            labels_by_split={"train": labels, "test": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",), eval_split="test", n_per_class=20, knn_ks=()
            ),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["n_folds"] == 1.0
        assert result.results["ridge"].metrics["balanced_accuracy_std"] == 0.0

    def test_skips_when_the_draw_would_leave_no_remainder(self) -> None:
        """A draw that consumes the pool declines rather than scoring on train.

        AEF bootstraps the eval split in that case (S4.4), which is a different
        protocol than the one implemented here.
        """
        labels = torch.arange(NUM_CLASSES).repeat(20)
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": _separable_embeddings(labels)},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",), n_per_class=20, knn_ks=()
            ),
            device=torch.device("cpu"),
        )
        assert result.results == {}

    def test_missing_test_split_shrinks_the_pool_rather_than_failing(self) -> None:
        """run_on_test=False degrades the pool to the available splits, not a crash."""
        embeddings, labels = self._splits()
        embeddings["test"] = None
        labels["test"] = None
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(n_folds=2, knn_ks=()),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["pool_size"] < 197.0
        assert result.results["ridge"].metrics["least_class"] < 12.0

    def test_fixed_eval_split_holds_the_eval_rows_constant(self) -> None:
        """A held-out eval split keeps the eval rows identical across folds."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(
                draw_pool=("train", "val"),
                eval_split="test",
                n_folds=3,
                knn_ks=(),
            ),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["eval_size"] == float(
            labels["test"].numel()
        )
        assert result.results["ridge"].metrics["pool_size"] == float(
            labels["train"].numel() + labels["val"].numel()
        )

    def test_rejects_an_eval_split_that_is_also_drawn_from(self) -> None:
        """Evaluating on a split that is also drawn from is refused, not silently run."""
        embeddings, labels = self._splits()
        with pytest.raises(ValueError, match="also in draw_pool"):
            run_balanced_trials(
                config=_config(),
                embeddings_by_split=embeddings,
                labels_by_split=labels,
                trial_config=BalancedTrialConfig(
                    draw_pool=("train", "val"), eval_split="val"
                ),
                device=torch.device("cpu"),
            )

    def test_ignores_out_of_range_labels(self) -> None:
        """Ignore-labelled rows are dropped before the draw."""
        labels = _imbalanced_labels()
        embeddings = _separable_embeddings(labels)
        labels = torch.cat([labels, torch.full((9,), -1, dtype=torch.long)])
        embeddings = torch.cat([embeddings, torch.zeros(9, EMBEDDING_DIM)])

        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": embeddings},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",), n_folds=2, knn_ks=()
            ),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["pool_size"] == 197.0

    def test_is_reproducible_across_runs(self) -> None:
        """Two runs of the same config produce identical metrics."""
        embeddings, labels = self._splits()
        trial_config = BalancedTrialConfig(n_folds=3, knn_ks=(5,))
        kwargs = dict(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=trial_config,
            device=torch.device("cpu"),
        )
        first = run_balanced_trials(**kwargs).results["ridge"].metrics
        second = run_balanced_trials(**kwargs).results["ridge"].metrics

        for key, value in first.items():
            assert second[key] == pytest.approx(value, nan_ok=True)

    def test_skips_a_single_class_pool(self) -> None:
        """A pool with one class has no balanced trial to run."""
        labels = torch.zeros(30, dtype=torch.long)
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": torch.randn(30, EMBEDDING_DIM)},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(draw_pool=("train",)),
            device=torch.device("cpu"),
        )
        assert result.results == {}

    def test_std_reflects_draw_to_draw_spread(self) -> None:
        """Std is the spread across draws; sem is that divided by sqrt(k)."""
        embeddings, labels = self._splits()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split=embeddings,
            labels_by_split=labels,
            trial_config=BalancedTrialConfig(n_folds=8, knn_ks=()),
            device=torch.device("cpu"),
        )
        per_fold = result.per_fold["ridge"]["balanced_accuracy"]
        assert len(per_fold) == 8
        mean = sum(per_fold) / len(per_fold)
        expected_std = math.sqrt(
            sum((v - mean) ** 2 for v in per_fold) / (len(per_fold) - 1)
        )
        assert result.results["ridge"].metrics["balanced_accuracy"] == pytest.approx(
            mean
        )
        assert result.results["ridge"].metrics[
            "balanced_accuracy_std"
        ] == pytest.approx(expected_std)
        # The error bar on the mean of k draws -- what AEF's figures carry --
        # is the draw-to-draw spread divided by sqrt(k).
        assert result.results["ridge"].metrics[
            "balanced_accuracy_sem"
        ] == pytest.approx(expected_std / math.sqrt(8))


class TestTrialTaskName:
    """The synthetic task name the trials report under."""

    def test_drops_the_host_knn_suffix_and_names_the_predictor(self) -> None:
        """A ridge result must not sit under a task name ending in _knn."""
        name = trial_task_name(
            "ethiopia_crops_year_aligned_ws16_ps1_sentinel2_knn", "ridge"
        )
        assert name == "ethiopia_crops_year_aligned_ws16_ps1_sentinel2_aeftrial_ridge"
        assert not name.endswith("_knn")

    def test_trial_knn_is_distinguishable_from_the_host_knn(self) -> None:
        """The whole point: the two kNN numbers cannot collide.

        Both are kNN at k=20 over the same embeddings, but one trains on the
        imbalanced train split and scores val while the other trains on a
        balanced draw and scores the remainder -- ~25 points apart on ethiopia.
        """
        host = "ethiopia_crops_year_aligned_ws16_ps1_sentinel2_knn"
        assert trial_task_name(host, "knn20") != host
        assert trial_task_name(host, "knn20").endswith("_aeftrial_knn20")

    def test_is_stable_for_a_host_without_a_knn_suffix(self) -> None:
        """Removesuffix is a no-op when the host is not the KNN twin."""
        assert trial_task_name("some_task", "ridge") == "some_task_aeftrial_ridge"


class TestLeastClassIsNotDrawnDry:
    """The rarest class must survive into the eval remainder."""

    def _pool(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Ethiopia-shaped: 4 classes, ~2.5k rows, least class 96."""
        counts = [1800, 400, 233, 96]
        labels = torch.cat(
            [torch.full((n,), i, dtype=torch.long) for i, n in enumerate(counts)]
        )
        return _separable_embeddings(labels), labels

    def test_draw_leaves_part_of_the_least_class_by_default(self) -> None:
        """The default never consumes a class whole, whatever the cap allows."""
        embeddings, labels = self._pool()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": embeddings},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",), n_folds=2, knn_ks=()
            ),
            device=torch.device("cpu"),
        )
        metrics = result.results["ridge"].metrics
        assert metrics["least_class"] == 96.0
        # int(96 * 0.9); the real sweeps set a per-dataset cap from AEF's
        # Table 1 that binds well below this.
        assert metrics["n_per_class"] == 86.0
        assert metrics["eval_classes"] == metrics["pool_classes"] == 4.0

    def test_every_class_survives_into_the_eval_set(self) -> None:
        """The failure this guards: a starved class silently shrinks the average.

        With n_per_class == least_class the rarest class is entirely consumed by
        the draw, and balanced accuracy is then averaged over K-1 classes
        without any indication in the number itself.
        """
        embeddings, labels = self._pool()
        kwargs = dict(
            config=_config(),
            embeddings_by_split={"train": embeddings},
            labels_by_split={"train": labels},
            device=torch.device("cpu"),
        )
        safe = (
            run_balanced_trials(
                trial_config=BalancedTrialConfig(
                    draw_pool=("train",), n_folds=2, knn_ks=()
                ),
                **kwargs,
            )
            .results["ridge"]
            .metrics
        )
        assert safe["eval_classes"] == safe["pool_classes"] == 4.0

        # The old behaviour, reachable only by asking for it explicitly.
        starved = (
            run_balanced_trials(
                trial_config=BalancedTrialConfig(
                    draw_pool=("train",), n_per_class=96, n_folds=2, knn_ks=()
                ),
                **kwargs,
            )
            .results["ridge"]
            .metrics
        )
        assert starved["eval_classes"] == 3.0
        assert starved["pool_classes"] == 4.0

    def test_fraction_is_configurable(self) -> None:
        """A caller who wants AEF-exact-per-their-table can set the fraction."""
        embeddings, labels = self._pool()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": embeddings},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",),
                least_class_draw_fraction=0.25,
                n_folds=2,
                knn_ks=(),
            ),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["n_per_class"] == 24.0

    def test_cap_still_binds_when_it_is_the_smaller(self) -> None:
        """The fraction only matters where the cap does not bind."""
        embeddings, labels = self._pool()
        result = run_balanced_trials(
            config=_config(),
            embeddings_by_split={"train": embeddings},
            labels_by_split={"train": labels},
            trial_config=BalancedTrialConfig(
                draw_pool=("train",), cap=10, n_folds=2, knn_ks=()
            ),
            device=torch.device("cpu"),
        )
        assert result.results["ridge"].metrics["n_per_class"] == 10.0
