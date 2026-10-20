"""RankTimingCallback records each rank's data wait and step time for cross-rank reduction."""

from types import SimpleNamespace

from olmo_core.train.common import ReduceType

from olmoearth_pretrain.train.callbacks.rank_timing import RankTimingCallback


def test_rank_timing_records_max_mean_and_negated_min() -> None:
    """Step time every step; data wait from the second step on; three reductions each."""
    recorded: list[tuple[str, float, ReduceType]] = []
    callback = RankTimingCallback()
    callback.trainer = SimpleNamespace(  # type: ignore[assignment]
        record_metric=lambda name, value, reduce: recorded.append((name, value, reduce))
    )
    callback.pre_train()
    for _ in range(2):
        callback.pre_step(None)
        callback.post_step()
    names = [name for name, _, _ in recorded]
    assert names.count("rank_timing/step_s_max") == 2
    assert names.count("rank_timing/data_wait_s_mean") == 1
    by_name = {name: (value, reduce) for name, value, reduce in recorded}
    assert by_name["rank_timing/step_s_max"][1] == ReduceType.max
    assert by_name["rank_timing/step_s_mean"][1] == ReduceType.mean
    value, reduce = by_name["rank_timing/step_s_negmin"]
    assert reduce == ReduceType.max and value <= 0
