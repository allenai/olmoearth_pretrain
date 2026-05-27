# OlmoEarth v1.2 Paper

LaTeX source for the OlmoEarth v1.2 technical report.

## Building

```bash
cd papers/v1_2
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

Or use `latexmk`:
```bash
latexmk -pdf main.tex
```

## Contents

- `main.tex` - Main paper source
- `references.bib` - Bibliography
- `README.md` - This file

## Key Changes from v1.1

OlmoEarth v1.2 replaces absolute sinusoidal position encodings with 2D Rotary Position Embeddings (RoPE):

1. **Axial 2D RoPE**: Standard rotary embeddings applied independently along row and column dimensions
2. **RoPE-Mixed**: Learnable 2D frequency vectors per attention head (Heo et al., 2024)

## Results Summary

Based on experiments from `henryh/2d-rope-v1-1-sweep` branch:
- RoPE-Mixed with base=10000 and coordinate_scale=0.25 shows best overall performance
- Consistent improvements on MADOS, PASTIS, and m-bigearthnet
- Better resolution generalization compared to absolute position encodings

## Notes

- Results tables contain placeholder values pending final experiment runs
- Update tables with actual numbers from wandb project `2026_04_22_add_hidden_layer_to_initial_projection`
