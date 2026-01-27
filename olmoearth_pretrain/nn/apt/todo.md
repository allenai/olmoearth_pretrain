1. Get this working for finetuning with some payoff
2. Get it to work for all bandsets for all modalities
2. Integrate into pretraining

# How do I appropriately set these thresholds? Need to do some experimentation and visualization to determine the best thresholds.
- we want to be able to visualize the tokenization for some subet of example and also compute how many tokens would be used for different tasks as well




# Fixes
- Ensure the band order selection in the scorer is correct


# THoughts
- do we need the patch descriptor class

# Done
- [x] Token counter/reduction logging in partitioner (use `log_stats=True` in `partition()` or `partition_temporal()`)
- [x] Logs: token count, uniform baseline, reduction %, tokens by scale
