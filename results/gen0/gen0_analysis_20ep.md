# Generation 0 analyses (runs\main_20261024\gen0_20ep, 100 completed networks, val only)

## 1. Medical val check

- Fitness ranking vs fitness without the 4 medical classes (11 classes): Spearman **0.997** (95% 0.992–0.998); best 20 shared: **18 / 20**
- Among the best 30 by fitness: Spearman **0.976** (95% 0.924–0.990); best 30 shared 30 / 30; their neighbouring fitness gaps median 0.10, medical val F1 spread among them 3.4 points
- Medical val F1 10 / 50 / 90%: 87.1 / 94.3 / 97.5; Spearman of medical val F1 with the 11-class fitness: 0.90
- Decision rule: a high Spearman keeps the fitness as it is; a low one is discussed.
- Limit: 11 of the 15 classes are shared and medical val saturates, so the overall Spearman is close to 1 almost by construction; the best-of rows are the informative ones. Memorizing the slides (medical val comes from the train slides) shows only on test, which this check cannot see.

## 2. Gene importance (target: val macro-F1, points)

Features with fewer than 5 networks on one side are left out. * = the 95% interval excludes 0.

### One feature at a time

| feature (any stage has it) | networks with it | val F1 with − without (95% CI) |
|---|---|---|
| act sigmoid | 17 | -12.06 (-22.97 … -3.82) * |
| norm bn | 60 | +6.27 (+2.25 … +10.37) * |
| head flatten | 23 | +6.20 (+2.10 … +10.24) * |
| act silu | 23 | +5.72 (+1.89 … +9.68) * |
| head gmp | 27 | -5.00 (-11.59 … +0.31) |
| no norm anywhere | 12 | -4.97 (-13.30 … +2.47) |
| act mish | 21 | +4.85 (+0.88 … +8.78) * |
| conn add | 55 | +4.83 (+0.59 … +9.68) * |
| act relu | 21 | +4.68 (+1.05 … +8.44) * |
| SE | 77 | +4.34 (+0.23 … +8.37) * |
| conn concat | 61 | +4.02 (-0.44 … +9.54) |
| downsample blurpool | 34 | +3.77 (-0.14 … +7.99) |
| body double_conv | 24 | +3.46 (-0.63 … +7.63) |
| downsample maxpool2 | 32 | +3.45 (-0.51 … +7.42) |
| act elu | 24 | -3.31 (-10.46 … +2.38) |
| act gelu | 26 | +3.16 (-0.57 … +7.00) |
| act identity | 23 | -3.02 (-10.77 … +2.82) |
| body bottleneck | 30 | +2.92 (-3.42 … +7.64) |
| norm ln | 57 | -2.43 (-6.83 … +1.73) |
| body conv | 41 | +2.38 (-1.62 … +6.78) |
| act tanh | 15 | +2.31 (-3.18 … +7.40) |
| act leaky_relu | 28 | +2.20 (-1.69 … +6.32) |
| valid padding | 5 | -1.67 (-5.49 … +2.38) |
| pre-activation | 54 | +1.31 (-3.33 … +5.12) |
| downsample avgpool | 36 | +1.10 (-3.13 … +5.33) |
| head gap | 26 | -1.06 (-6.33 … +3.73) |
| body dw_sep | 40 | +1.01 (-4.11 … +5.47) |
| body mbconv | 39 | -0.87 (-5.93 … +3.44) |
| downsample strided_conv | 36 | -0.85 (-7.02 … +4.13) |
| act hardswish | 22 | +0.81 (-7.70 … +7.00) |
| body convnext | 38 | -0.60 (-5.82 … +3.71) |
| downsample maxpool3 | 28 | -0.53 (-7.15 … +4.71) |
| head gap_gmp | 24 | +0.50 (-3.50 … +4.54) |

| numeric feature | Spearman with val F1 (95% CI) |
|---|---|
| log2 params | +0.47 (+0.30 … +0.62) * |
| log2 MACs | +0.46 (+0.29 … +0.61) * |
| log2 max width | +0.40 (+0.20 … +0.57) * |
| stages | +0.40 (+0.21 … +0.56) * |
| log2 first slot | +0.38 (+0.20 … +0.54) * |
| total depth | +0.37 (+0.18 … +0.53) * |
| head FC layers | -0.11 (-0.29 … +0.07) |
| stem width | +0.05 (-0.15 … +0.25) |

### All together: ridge regression (λ = 100, the lowest leave-one-out error; leave-one-out R² 0.23, optimistic since λ was chosen on it)

Coefficient = val F1 points per one standard deviation of the feature, the others held fixed. With log2 params / MACs among the features this is a gene's effect beyond the size it brings (a direct effect, not the total one), and none of it is causal. Many intervals at once: a few * are expected by chance.

| feature | coefficient (95% CI) |
|---|---|
| act sigmoid | -2.06 (-3.19 … -0.67) * |
| conn add | +1.20 (+0.07 … +2.15) * |
| log2 max width | +0.97 (+0.24 … +1.55) * |
| head FC layers | -0.95 (-1.48 … -0.23) * |
| act silu | +0.91 (+0.30 … +1.46) * |
| norm ln | -0.86 (-1.34 … -0.27) * |
| act identity | -0.85 (-1.72 … +0.03) |
| act elu | -0.83 (-1.80 … +0.17) |
| log2 params | +0.76 (+0.17 … +1.27) * |
| head gmp | -0.74 (-1.34 … -0.13) * |
| head flatten | +0.71 (+0.11 … +1.22) * |
| conn concat | +0.70 (-0.17 … +1.50) |
| body convnext | -0.66 (-1.34 … +0.13) |
| log2 first slot | +0.63 (+0.13 … +1.17) * |
| norm bn | +0.63 (-0.02 … +1.23) |
| act relu | +0.61 (+0.10 … +1.03) * |
| log2 MACs | +0.56 (-0.07 … +1.06) |
| downsample maxpool2 | +0.55 (-0.13 … +1.17) |
| no norm anywhere | -0.54 (-1.23 … +0.23) |
| act gelu | +0.52 (-0.12 … +1.08) |
| act mish | +0.52 (-0.04 … +1.02) |
| total depth | +0.51 (+0.06 … +0.99) * |
| downsample strided_conv | -0.48 (-1.24 … +0.32) |
| downsample blurpool | +0.43 (-0.24 … +1.11) |
| SE | +0.37 (-0.19 … +0.91) |
| downsample maxpool3 | -0.33 (-1.03 … +0.38) |
| stem width | +0.30 (-0.22 … +0.79) |
| body mbconv | -0.26 (-1.07 … +0.53) |
| stages | +0.19 (-0.37 … +0.67) |
| body dw_sep | +0.17 (-0.34 … +0.62) |
| act tanh | +0.12 (-0.53 … +0.77) |
| head gap_gmp | +0.11 (-0.38 … +0.60) |
| downsample avgpool | +0.10 (-0.55 … +0.79) |
| body conv | +0.10 (-0.46 … +0.76) |
| act hardswish | -0.08 (-0.92 … +0.68) |
| act leaky_relu | +0.07 (-0.43 … +0.65) |
| body bottleneck | -0.07 (-1.08 … +0.84) |
| pre-activation | +0.06 (-0.51 … +0.62) |
| head gap | -0.04 (-0.64 … +0.52) |
| valid padding | +0.03 (-0.48 … +0.64) |
| body double_conv | +0.02 (-0.45 … +0.60) |
