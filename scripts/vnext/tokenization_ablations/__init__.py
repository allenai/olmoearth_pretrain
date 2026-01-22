"""Tokenization ablation experiments.

This folder contains experiments testing different bandset tokenization strategies
using the TokenizationConfig feature. All experiments are based on base.py and only
differ in how bands are grouped into tokens.

Experiments:
- base_single_band_s2.py: Each Sentinel-2 band as its own token (12 tokens)
- base_spectral_grouping.py: Bands grouped by spectral similarity (5 groups)
- base_all_bands_single_token.py: All bands in one token (1 token per modality)
"""
