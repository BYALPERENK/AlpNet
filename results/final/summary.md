# Search networks retrained (5 seeds, the baselines' recipe) vs baselines

## Checkpoint: best val macro_f1 — test scores (mean ± std over seeds, %)

| Model | Params | MACs | train time / run | epoch (of 40) | val macro_f1 | test acc | test macro-F1 | test AUROC | test cross-group err | test medical | test animal | test vehicle | test scene |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1_lenet | 0.34M | 0.00G | 0.5 ± 0.1 min | 38, 36, 37 | 75.76 ± 0.90 | 74.44 ± 1.22 | 74.22 ± 1.24 | 97.12 ± 0.22 | 7.71 ± 0.63 | 88.74 ± 1.15 | 93.57 ± 1.35 | 58.70 ± 1.70 | 60.69 ± 2.02 |
| b2_plain | 1.18M | 0.21G | 0.7 ± 0.0 min | 39, 36, 38 | 88.18 ± 0.46 | 86.52 ± 0.15 | 86.51 ± 0.14 | 98.99 ± 0.02 | 2.89 ± 0.20 | 94.88 ± 0.43 | 98.67 ± 0.28 | 78.80 ± 0.63 | 76.70 ± 0.41 |
| b3_resnet18 | 11.18M | 2.22G | 3.1 ± 0.1 min | 35, 38, 37 | 89.04 ± 0.09 | 87.60 ± 0.05 | 87.61 ± 0.06 | 98.76 ± 0.09 | 2.47 ± 0.14 | 95.26 ± 0.36 | 98.88 ± 0.25 | 81.30 ± 0.50 | 77.83 ± 0.89 |
| b4_mobilevitv2 | 13.37M | 1.37G | 6.4 ± 0.3 min | 37, 38, 39 | 88.58 ± 0.20 | 87.20 ± 0.37 | 87.20 ± 0.36 | 98.82 ± 0.10 | 2.76 ± 0.16 | 95.37 ± 0.69 | 98.93 ± 0.15 | 79.85 ± 0.45 | 77.56 ± 0.36 |
| alpnet_a | 0.62M | 0.30G | 3.8 ± 0.4 min | 40, 37, 37, 36, 37 | 88.77 ± 0.12 | 86.94 ± 0.20 | 86.93 ± 0.20 | 99.11 ± 0.04 | 2.49 ± 0.13 | 95.10 ± 0.44 | 98.71 ± 0.20 | 79.33 ± 0.20 | 77.51 ± 0.62 |
| alpnet_b | 0.30M | 0.18G | 3.6 ± 0.3 min | 38, 38, 36, 39, 38 | 87.75 ± 0.15 | 86.08 ± 0.11 | 86.07 ± 0.11 | 99.13 ± 0.02 | 2.58 ± 0.13 | 94.71 ± 0.38 | 98.10 ± 0.27 | 78.20 ± 0.54 | 76.26 ± 0.32 |
| alpnet_c | 0.43M | 0.23G | 3.8 ± 0.1 min | 38, 38, 38, 37, 40 | 88.21 ± 0.14 | 86.46 ± 0.27 | 86.44 ± 0.27 | 99.03 ± 0.04 | 2.77 ± 0.17 | 94.98 ± 0.43 | 98.50 ± 0.16 | 78.47 ± 0.26 | 76.82 ± 0.96 |
| alpnet_d | 0.52M | 0.32G | 4.9 ± 0.2 min | 36, 40, 40, 35, 38 | 88.46 ± 0.27 | 86.90 ± 0.24 | 86.90 ± 0.25 | 99.00 ± 0.05 | 2.31 ± 0.32 | 94.91 ± 0.42 | 98.75 ± 0.25 | 79.06 ± 0.84 | 77.84 ± 0.20 |
| alpnet_e | 0.30M | 0.18G | 4.1 ± 0.1 min | 39, 40, 40, 36, 38 | 87.82 ± 0.27 | 86.09 ± 0.13 | 86.08 ± 0.13 | 99.14 ± 0.02 | 2.55 ± 0.08 | 95.00 ± 0.44 | 98.20 ± 0.28 | 77.76 ± 0.66 | 76.38 ± 0.53 |
| alpnet_f | 0.45M | 0.06G | 3.5 ± 0.0 min | 37, 39, 38, 38, 36 | 86.87 ± 0.24 | 85.23 ± 0.21 | 85.19 ± 0.19 | 98.94 ± 0.02 | 3.17 ± 0.11 | 94.73 ± 0.26 | 98.34 ± 0.37 | 76.21 ± 0.48 | 74.78 ± 0.65 |
| alpnet_g | 0.80M | 0.55G | 7.4 ± 0.1 min | 37, 40, 30, 40, 35 | 88.74 ± 0.24 | 86.83 ± 0.23 | 86.84 ± 0.24 | 98.90 ± 0.07 | 2.25 ± 0.04 | 94.30 ± 0.64 | 98.68 ± 0.12 | 79.72 ± 0.43 | 77.60 ± 0.30 |

Networks (search: one training, seed 0, the search recipe; chosen by val):

- alpnet_a: f26817817324fcc9, generation 18, search val F1 88.81, fitness 88.55: s64 mbconv/concat d1 w96 k1 bn | s32 bottleneck/add d2 w128 k3 ln | s16 conv/concat d5 w64 k3 bn | s4 double_conv/concat d2 w96 k1 bn
- alpnet_b: c00c957e623f0814, generation 20, search val F1 88.21, fitness 88.47: s64 mbconv/concat d1 w128 k3 bn | s32 bottleneck/add d6 w48 k3 none | s16 double_conv/concat d4 w64 k3 bn | s4 double_conv/add d2 w192 k1 bn
- alpnet_c: e343de5ba1d4ce41, generation 16, search val F1 88.33, fitness 88.32: s64 mbconv/concat d1 w96 k1 bn | s32 bottleneck/add d1 w128 k3 ln | s16 conv/concat d5 w64 k3 bn | s4 double_conv/concat d2 w192 k1 bn
- alpnet_d: 2505f5933234422d, generation 18, search val F1 88.64, fitness 88.42: s64 mbconv/concat d1 w256 k3 bn | s32 bottleneck/add d2 w128 k3 bn | s8 convnext/concat d4 w256 k5 bn | s4 double_conv/concat d3 w192 k1 bn
- alpnet_e: 592e482d4996cc42, generation 19, search val F1 88.08, fitness 88.33: s64 mbconv/concat d1 w128 k3 bn | s32 bottleneck/add d6 w48 k3 none | s16 double_conv/concat d4 w64 k3 bn | s4 double_conv/add d2 w192 k1 bn
- alpnet_f: f2ffd912f0fafd0e, generation 16, search val F1 86.70, fitness 87.27: s32 bottleneck/concat d1 w128 k3 bn | s16 conv/concat d4 w256 k3 bn | s8 convnext/concat d1 w96 k1 bn | s4 double_conv/concat d3 w192 k1 none
- alpnet_g: 5471c6b4e462f23a, generation 16, search val F1 88.82, fitness 88.17: s64 mbconv/add d3 w96 k3 bn | s16 conv/concat d6 w384 k3 ln | s4 bottleneck/concat d2 w192 k1 bn

## Selection optimism (winner's curse)

Search = the one training the network was chosen by (seed 0, the search recipe); its test score was sealed during the search. Retrained = mean ± std over the seeds above. Δ = retrained − search.

| network | search val F1 | retrained val F1 | Δ val | search test F1 | retrained test F1 | Δ test |
|---|---|---|---|---|---|---|
| alpnet_a | 88.81 | 88.77 ± 0.12 | -0.04 | 87.21 | 86.93 ± 0.20 | -0.28 |
| alpnet_b | 88.21 | 87.75 ± 0.15 | -0.46 | 86.03 | 86.07 ± 0.11 | +0.03 |
| alpnet_c | 88.33 | 88.21 ± 0.14 | -0.12 | 86.31 | 86.44 ± 0.27 | +0.13 |
| alpnet_d | 88.64 | 88.46 ± 0.27 | -0.17 | 87.46 | 86.90 ± 0.25 | -0.56 |
| alpnet_e | 88.08 | 87.82 ± 0.27 | -0.26 | 86.50 | 86.08 ± 0.13 | -0.43 |
| alpnet_f | 86.70 | 86.87 ± 0.24 | +0.16 | 85.53 | 85.19 ± 0.19 | -0.33 |
| alpnet_g | 88.82 | 88.74 ± 0.24 | -0.08 | 85.85 | 86.84 ± 0.24 | +0.98 |
