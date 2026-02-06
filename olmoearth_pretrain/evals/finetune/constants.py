"""Constants for finetuning."""

# Fraction of total epochs to keep backbone frozen before unfreezing.
FREEZE_EPOCH_FRACTION = 0.2

# Factor to multiply learning rate by when unfreezing backbone (e.g., 0.1 = reduce LR by 10x).
UNFREEZE_LR_FACTOR = 0.1

# Factor by which to reduce learning rate when validation metric plateaus.
SCHEDULER_FACTOR = 0.2

# Number of epochs with no improvement before reducing learning rate.
SCHEDULER_PATIENCE = 2

# Minimum learning rate the scheduler can reduce to.
SCHEDULER_MIN_LR = 0.0

# Number of epochs to wait after a LR reduction before resuming normal operation.
SCHEDULER_COOLDOWN = 10
