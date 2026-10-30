# Generation 0 analyses (runs\main_20261024\gen0_40ep, 100 completed networks, val only)

## 1. Medical val check

- Fitness ranking vs fitness without the 4 medical classes (11 classes): Spearman **0.997** (95% 0.992–0.998); best 20 shared: **19 / 20**
- Among the best 30 by fitness: Spearman **0.968** (95% 0.904–0.991); best 30 shared 30 / 30; their neighbouring fitness gaps median 0.08, medical val F1 spread among them 2.0 points
- Medical val F1 10 / 50 / 90%: 90.4 / 96.1 / 98.0; Spearman of medical val F1 with the 11-class fitness: 0.91
- Decision rule: a high Spearman keeps the fitness as it is; a low one is discussed.
- Limit: 11 of the 15 classes are shared and medical val saturates, so the overall Spearman is close to 1 almost by construction; the best-of rows are the informative ones. Memorizing the slides (medical val comes from the train slides) shows only on test, which this check cannot see.

## 2. Gene importance (target: val macro-F1, points)

Features with fewer than 5 networks on one side are left out. * = the 95% interval excludes 0.

### One feature at a time

| feature (any stage has it) | networks with it | val F1 with − without (95% CI) |
|---|---|---|
| act sigmoid | 17 | -10.14 (-21.20 … -2.14) * |
| head flatten | 23 | +5.22 (+1.80 … +8.85) * |
| no norm anywhere | 12 | -5.11 (-12.66 … +1.58) |
| act mish | 21 | +4.94 (+1.70 … +8.27) * |
| norm bn | 60 | +4.91 (+1.07 … +8.73) * |
| act silu | 23 | +4.79 (+1.28 … +8.30) * |
| conn concat | 61 | +4.68 (+0.45 … +10.13) * |
| act relu | 21 | +4.21 (+0.84 … +7.75) * |
| conn add | 55 | +4.15 (+0.23 … +8.55) * |
| SE | 77 | +3.98 (+0.19 … +7.60) * |
| head gmp | 27 | -3.91 (-10.79 … +1.11) |
| body double_conv | 24 | +3.78 (+0.46 … +7.43) * |
| downsample maxpool2 | 32 | +3.75 (+0.39 … +7.37) * |
| downsample blurpool | 34 | +3.69 (+0.36 … +7.32) * |
| act identity | 23 | -3.63 (-11.53 … +1.97) |
| act gelu | 26 | +3.27 (+0.14 … +6.71) * |
| act elu | 24 | -2.69 (-10.04 … +2.70) |
| act tanh | 15 | +2.59 (-1.56 … +6.72) |
| act leaky_relu | 28 | +2.24 (-1.24 … +5.89) |
| body conv | 41 | +2.11 (-1.48 … +6.22) |
| body bottleneck | 30 | +1.80 (-4.47 … +6.30) |
| body dw_sep | 40 | +1.49 (-3.55 … +5.44) |
| head gap | 26 | -1.42 (-6.15 … +2.88) |
| valid padding | 5 | -1.39 (-4.52 … +2.06) |
| pre-activation | 54 | +1.35 (-2.96 … +4.79) |
| downsample avgpool | 36 | +1.23 (-2.38 … +5.15) |
| downsample maxpool3 | 28 | -1.15 (-7.83 … +3.88) |
| norm ln | 57 | -1.09 (-5.36 … +2.64) |
| body mbconv | 39 | -1.07 (-5.93 … +3.03) |
| head gap_gmp | 24 | +0.65 (-2.78 … +4.27) |
| body convnext | 38 | -0.40 (-5.64 … +3.54) |
| downsample strided_conv | 36 | -0.36 (-6.44 … +4.15) |
| act hardswish | 22 | +0.08 (-8.37 … +5.83) |

| numeric feature | Spearman with val F1 (95% CI) |
|---|---|
| log2 MACs | +0.53 (+0.36 … +0.67) * |
| log2 params | +0.50 (+0.33 … +0.64) * |
| stages | +0.45 (+0.26 … +0.60) * |
| total depth | +0.43 (+0.24 … +0.59) * |
| log2 first slot | +0.41 (+0.23 … +0.57) * |
| log2 max width | +0.39 (+0.19 … +0.58) * |
| stem width | +0.07 (-0.12 … +0.27) |
| head FC layers | -0.04 (-0.22 … +0.15) |

### All together: ridge regression (λ = 100, the lowest leave-one-out error; leave-one-out R² 0.16, optimistic since λ was chosen on it)

Coefficient = val F1 points per one standard deviation of the feature, the others held fixed. With log2 params / MACs among the features this is a gene's effect beyond the size it brings (a direct effect, not the total one), and none of it is causal. Many intervals at once: a few * are expected by chance.

| feature | coefficient (95% CI) |
|---|---|
| act sigmoid | -1.78 (-2.92 … -0.48) * |
| conn add | +1.06 (-0.01 … +2.03) |
| act identity | -0.96 (-1.86 … -0.09) * |
| conn concat | +0.85 (+0.01 … +1.68) * |
| act silu | +0.77 (+0.21 … +1.31) * |
| log2 max width | +0.73 (+0.11 … +1.24) * |
| act elu | -0.71 (-1.75 … +0.27) |
| head FC layers | -0.69 (-1.25 … -0.06) * |
| downsample maxpool2 | +0.64 (+0.01 … +1.24) * |
| log2 first slot | +0.62 (+0.13 … +1.12) * |
| log2 MACs | +0.62 (-0.01 … +1.08) |
| head flatten | +0.58 (+0.09 … +1.06) * |
| norm ln | -0.57 (-1.08 … -0.04) * |
| body convnext | -0.57 (-1.28 … +0.17) |
| act mish | +0.55 (+0.07 … +1.06) * |
| no norm anywhere | -0.55 (-1.15 … +0.15) |
| act relu | +0.54 (+0.07 … +0.97) * |
| act gelu | +0.54 (-0.05 … +1.13) |
| log2 params | +0.53 (-0.04 … +1.02) |
| total depth | +0.53 (+0.12 … +0.98) * |
| head gmp | -0.53 (-1.16 … +0.05) |
| downsample blurpool | +0.45 (-0.13 … +1.08) |
| downsample maxpool3 | -0.44 (-1.17 … +0.25) |
| norm bn | +0.38 (-0.26 … +0.95) |
| downsample strided_conv | -0.36 (-1.10 … +0.36) |
| SE | +0.33 (-0.19 … +0.83) |
| body mbconv | -0.32 (-1.12 … +0.46) |
| stem width | +0.28 (-0.15 … +0.72) |
| body dw_sep | +0.28 (-0.19 … +0.66) |
| body bottleneck | -0.27 (-1.31 … +0.65) |
| act tanh | +0.25 (-0.33 … +0.84) |
| act hardswish | -0.23 (-1.08 … +0.51) |
| downsample avgpool | +0.14 (-0.46 … +0.82) |
| head gap | -0.13 (-0.66 … +0.40) |
| stages | +0.12 (-0.44 … +0.60) |
| body double_conv | +0.11 (-0.29 … +0.64) |
| act leaky_relu | +0.11 (-0.35 … +0.66) |
| head gap_gmp | +0.11 (-0.32 … +0.55) |
| valid padding | +0.10 (-0.38 … +0.70) |
| body conv | +0.07 (-0.44 … +0.70) |
| pre-activation | +0.03 (-0.50 … +0.53) |
