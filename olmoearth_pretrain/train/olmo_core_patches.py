"""Runtime patches for olmo_core bugs that affect our training runs.

``drop_async_checkpoint_duration_metric``: olmo_core 2.6.0's
``Trainer.save_checkpoint_async`` records ``checkpoint/save_async_duration_s`` from the
async-save done-callback, i.e. from a background thread, at whatever step the main
thread is on when that rank's save finishes. Ranks finish at different times, so one
rank can flush the metric in its ``reduce_metrics`` call while another flushes it in
the next one. With ``metrics_consistent=True`` the ranks then all-reduce tensors of
different shapes on the gloo bookkeeping group, and the bookkeeping thread hangs. The
next checkpoint save blocks forever in ``_join_bookkeeping_ops`` ("Waiting for
bookkeeping ops to finish"), or gloo times out ~30 min later with "Application
timeout caused pair closure". Rank skew makes this likely: e.g. rank 0 launching an
in-loop Beaker eval in ``post_step`` right after the async save of an eval step starts,
or the extra metric flush at an epoch end. The v1.3 NISAR runs hit it five times in
~2 days. Dropping the metric removes the rank-dependent record; the synchronous
``checkpoint/save_duration_s`` (recorded on the main thread) is unaffected.
``internal.experiment.train`` applies this patch before building the trainer.
"""

from olmo_core.train.trainer import Trainer

ASYNC_CHECKPOINT_DURATION_METRIC = "checkpoint/save_async_duration_s"


def drop_async_checkpoint_duration_metric() -> None:
    """Make ``Trainer.record_metric`` ignore the async-save duration metric (idempotent)."""
    if getattr(Trainer.record_metric, "_drops_async_ckpt_metric", False) is True:
        return
    original = Trainer.record_metric

    def record_metric(self, name, value, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == ASYNC_CHECKPOINT_DURATION_METRIC:
            return
        return original(self, name, value, *args, **kwargs)

    record_metric._drops_async_ckpt_metric = True  # type: ignore[attr-defined]
    Trainer.record_metric = record_metric  # type: ignore[method-assign]
