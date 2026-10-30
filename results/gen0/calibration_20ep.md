# Stop rule calibration on 100 fully trained candidates (runs\main_20261024\gen0_20ep)

B's elite bar (K = 20-th best fitness of the others) ~ 80.38; protected: the best 30 (bar ~ 79.08, 1.30 points below the elite bar); fitness 10 / 50 / 90%: 63.0 / 74.5 / 81.8

Chosen setting per rule: the one saving the most with no wrong cut (ties: the tighter).

## Fixed rules

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| not learning | 1 | 1 | 0.8% | 1.4% | 0 | -0.1 |
| patience | 0 | 22 | 3.5% | 3.1% | 0 | nan |

## D floor

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| epoch 5 < 0.40 | 12 | 12 | 9.0% | 10.8% | 1 | 80.5 |
| epoch 5 < 0.45 | 17 | 17 | 12.8% | 14.7% | 1 | 80.5 |
| epoch 5 < 0.50 | 25 | 25 | 18.8% | 19.8% | 1 | 80.5 |
| epoch 5 < 0.55 | 34 | 34 | 25.5% | 24.6% | 1 | 80.5 |
| epoch 5 < 0.60 | 51 | 51 | 38.2% | 34.4% | 4 | 80.5 |
| epoch 5 < 0.65 | 69 | 69 | 51.8% | 49.0% | 7 | 83.6 |
| epoch 5 < 0.70 | 87 | 87 | 65.2% | 61.6% | 18 | 83.6 |

Chosen: **none** (every setting cuts a protected candidate)

## A percentile

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| 10% at 10 | 11 | 11 | 5.5% | 5.7% | 0 | 73.3 |
| 20% at 10 | 21 | 21 | 10.5% | 9.9% | 0 | 73.3 |
| 30% at 10 | 31 | 31 | 15.5% | 13.5% | 1 | 79.1 |
| 50% at 10 | 50 | 50 | 25.0% | 22.2% | 3 | 80.5 |
| 5% / 10% at 5 / 10 | 13 | 13 | 8.0% | 9.7% | 1 | 80.5 |
| 10% / 20% at 5 / 10 | 22 | 22 | 13.8% | 14.7% | 1 | 80.5 |
| 20% / 30% at 5 / 10 | 32 | 32 | 21.2% | 20.6% | 2 | 80.5 |
| 30% / 50% at 5 / 10 | 50 | 50 | 32.8% | 29.9% | 3 | 80.5 |
| 40% / 60% at 5 / 10 | 60 | 60 | 40.0% | 36.9% | 3 | 80.5 |
| 5% / 10% / 15% at 2 / 5 / 10 | 17 | 17 | 12.0% | 13.3% | 1 | 80.5 |
| 10% / 15% / 25% at 2 / 5 / 10 | 29 | 29 | 20.6% | 20.8% | 2 | 80.5 |
| 20% / 30% / 50% at 2 / 5 / 10 | 52 | 52 | 37.6% | 36.3% | 4 | 80.5 |
| 30% / 40% / 60% at 2 / 5 / 10 | 62 | 62 | 46.5% | 46.0% | 5 | 83.6 |
| 40% / 50% / 70% at 2 / 5 / 10 | 73 | 73 | 56.5% | 54.9% | 9 | 83.6 |

Chosen: **20% at 10**

## B reach_elite

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| q 1.0 at 15 | 53 | 53 | 13.2% | 11.0% | 0 | 77.9 |
| q 0.95 at 15 | 66 | 66 | 16.5% | 14.9% | 3 | 80.5 |
| q 0.9 at 15 | 72 | 72 | 18.0% | 16.3% | 4 | 80.5 |
| q 0.8 at 15 | 77 | 77 | 19.2% | 17.4% | 7 | 80.5 |
| q 1.0 at 10/15 | 53 | 53 | 16.5% | 14.5% | 0 | 77.9 |
| q 0.95 at 10/15 | 66 | 66 | 26.0% | 23.4% | 3 | 80.5 |
| q 0.9 at 10/15 | 72 | 72 | 31.0% | 27.8% | 4 | 80.5 |
| q 0.8 at 10/15 | 78 | 78 | 36.0% | 33.1% | 8 | 80.5 |
| q 1.0 at 5/10/15 | 54 | 54 | 17.8% | 16.8% | 1 | 80.5 |
| q 0.95 at 5/10/15 | 66 | 66 | 33.0% | 30.9% | 3 | 80.5 |
| q 0.9 at 5/10/15 | 72 | 72 | 40.0% | 36.4% | 4 | 80.5 |
| q 0.8 at 5/10/15 | 79 | 79 | 51.2% | 48.4% | 9 | 83.6 |
| q 1.0 at 2/5/10/15 | 54 | 54 | 20.7% | 21.0% | 1 | 80.5 |
| q 0.95 at 2/5/10/15 | 66 | 66 | 36.6% | 35.2% | 3 | 80.5 |
| q 0.9 at 2/5/10/15 | 74 | 74 | 46.4% | 45.5% | 6 | 83.6 |
| q 0.8 at 2/5/10/15 | 80 | 80 | 59.2% | 57.8% | 10 | 83.6 |

Chosen: **q 1.0 at 10/15**

## Vetoes: protected candidates the settings cut

A rule whose every setting cuts one of these is off; decide knowingly whether that network is worth it.

| network | fitness (rank) | best val F1 by epoch 2 / 5 / 10 → end | cut by | architecture |
|---|---|---|---|---|
| 026_883cc5048b2be7cc | 83.64 (1) | 31.9 / 61.2 / 75.5 → 84.6 | D 2 of 7 (loosest: epoch 5 < 0.65); A 2 of 14 (loosest: 30% / 40% / 60% at 2 / 5 / 10); B 3 of 16 (loosest: q 0.8 at 5/10/15) | s64 conv/none d6 w128 k1 ln | s32 double_conv/add d2 w32 k3 bn | s16 convnext/none d4 w256 k3 none | s8 bottleneck/concat d6 w96 k7 bn | s4 dw_sep/none d3 w96 k1 ln |
| 048_c2cdf0365f464cde | 82.90 (2) | 62.4 / 69.5 / 77.3 → 82.8 | D 1 of 7 (loosest: epoch 5 < 0.70) | s64 mbconv/concat d1 w48 k5 bn | s16 conv/concat d4 w96 k3 ln | s8 convnext/concat d6 w64 k1 bn | s4 double_conv/concat d2 w24 k1 bn |
| 098_647c679ce070da30 | 82.22 (7) | 48.8 / 69.3 / 79.1 → 83.4 | D 1 of 7 (loosest: epoch 5 < 0.70) | s16 convnext/add d4 w384 k3 bn | s8 double_conv/concat d3 w16 k1 bn |
| 052_d1e5813b85817ed7 | 81.61 (13) | 50.5 / 68.4 / 77.3 → 81.1 | D 1 of 7 (loosest: epoch 5 < 0.70) | s8 bottleneck/add d5 w64 k7 bn |
| 069_eaf7133434175c39 | 80.52 (16) | 26.8 / 26.8 / 66.1 → 81.4 | D 7 of 7 (loosest: epoch 5 < 0.40); A 11 of 14 (loosest: 50% at 10); B 14 of 16 (loosest: q 0.95 at 15) | s64 mbconv/none d4 w128 k3 bn | s4 mbconv/none d2 w256 k3 none |
| 022_fe0d75accfc73221 | 80.48 (17) | 53.7 / 67.6 / 75.5 → 81.4 | D 1 of 7 (loosest: epoch 5 < 0.70) | s64 mbconv/add d4 w24 k1 ln | s32 convnext/add d2 w48 k1 ln | s16 double_conv/concat d6 w96 k7 ln |
| 039_cc209982537128d0 | 80.45 (18) | 53.0 / 69.4 / 76.9 → 81.0 | D 1 of 7 (loosest: epoch 5 < 0.70) | s32 dw_sep/none d1 w192 k5 bn | s16 conv/concat d2 w128 k7 bn |
| 036_ff91753454fdecaa | 80.42 (19) | 55.9 / 67.2 / 75.7 → 79.8 | D 1 of 7 (loosest: epoch 5 < 0.70) | s64 double_conv/add d6 w32 k1 none | s16 convnext/add d2 w64 k7 bn | s8 bottleneck/add d6 w48 k7 none | s4 double_conv/concat d4 w192 k1 ln |
| 078_eb14f6786eecba6f | 80.38 (20) | 48.2 / 67.6 / 73.5 → 79.9 | D 1 of 7 (loosest: epoch 5 < 0.70) | s32 bottleneck/concat d5 w384 k5 bn |
| 076_fbe32e4cbb0a0bda | 80.20 (21) | 42.3 / 65.4 / 72.2 → 80.5 | D 1 of 7 (loosest: epoch 5 < 0.70); A 1 of 14 (loosest: 40% / 50% / 70% at 2 / 5 / 10); B 3 of 16 (loosest: q 0.8 at 10/15) | s64 mbconv/concat d2 w256 k3 none | s16 double_conv/concat d2 w48 k7 bn |
| 003_8f1d788ca404b00e | 80.01 (22) | 36.8 / 67.3 / 73.5 → 79.2 | D 1 of 7 (loosest: epoch 5 < 0.70); A 1 of 14 (loosest: 40% / 50% / 70% at 2 / 5 / 10); B 1 of 16 (loosest: q 0.8 at 2/5/10/15) | s64 dw_sep/concat d3 w64 k1 ln | s16 mbconv/none d3 w128 k3 ln | s4 dw_sep/concat d2 w48 k1 none |
| 070_51334fcc56213f08 | 79.92 (23) | 52.8 / 62.4 / 74.2 → 80.0 | D 2 of 7 (loosest: epoch 5 < 0.65); B 4 of 16 (loosest: q 0.8 at 15) | s32 double_conv/concat d4 w96 k5 ln | s16 bottleneck/concat d3 w192 k3 bn | s4 conv/add d4 w128 k3 none |
| 006_7c8e86dd2db8e21b | 79.83 (24) | 53.3 / 65.7 / 74.3 → 78.8 | D 1 of 7 (loosest: epoch 5 < 0.70) | s64 mbconv/none d1 w48 k5 bn | s32 bottleneck/none d1 w96 k7 ln | s16 dw_sep/none d2 w24 k5 bn | s8 convnext/concat d4 w96 k5 none | s4 dw_sep/add d2 w64 k1 ln |
| 094_ff9aca87dcbea2cd | 79.33 (26) | 28.4 / 64.8 / 73.9 → 79.9 | D 2 of 7 (loosest: epoch 5 < 0.65); A 3 of 14 (loosest: 20% / 30% / 50% at 2 / 5 / 10); B 5 of 16 (loosest: q 0.8 at 15) | s64 dw_sep/add d2 w96 k7 ln | s32 dw_sep/none d2 w256 k1 none | s16 dw_sep/none d6 w256 k5 none | s4 mbconv/add d3 w96 k3 none |
| 027_559cc8fdfee71fbf | 79.18 (27) | 38.2 / 57.7 / 72.4 → 78.1 | D 3 of 7 (loosest: epoch 5 < 0.60); A 1 of 14 (loosest: 40% / 50% / 70% at 2 / 5 / 10); B 4 of 16 (loosest: q 0.8 at 15) | s64 convnext/none d3 w24 k3 ln | s4 dw_sep/add d5 w96 k7 bn |
| 071_b2e591aff486b1b1 | 79.16 (28) | 40.5 / 58.3 / 72.3 → 79.1 | D 3 of 7 (loosest: epoch 5 < 0.60); A 1 of 14 (loosest: 40% / 50% / 70% at 2 / 5 / 10); B 8 of 16 (loosest: q 0.9 at 15) | s16 conv/none d5 w64 k3 none | s8 convnext/none d5 w128 k1 bn | s4 bottleneck/none d1 w128 k3 none |
| 042_932878fabd40b9a7 | 79.10 (29) | 32.3 / 55.4 / 58.3 → 78.7 | D 3 of 7 (loosest: epoch 5 < 0.60); A 9 of 14 (loosest: 30% at 10); B 12 of 16 (loosest: q 0.95 at 15) | s32 bottleneck/add d1 w192 k3 ln | s16 dw_sep/concat d3 w48 k1 ln | s8 conv/add d6 w48 k3 bn |
| 074_1adaa162f4e6c832 | 79.08 (30) | 36.1 / 65.2 / 66.8 → 79.9 | D 1 of 7 (loosest: epoch 5 < 0.70); A 6 of 14 (loosest: 50% at 10); B 12 of 16 (loosest: q 0.95 at 15) | s32 dw_sep/concat d4 w384 k3 ln | s16 convnext/none d4 w96 k1 bn | s4 conv/add d5 w128 k5 ln |

## All chosen rules together

- Stopped 64 / 100 ({'percentile': 20, 'reach_elite': 32, 'patience': 11, 'not learning': 1}); time saved **17.0%**, wrong cuts 0 (B saves more in the GA: its bar rises)

## C: successive halving (epoch 5 → best half → epoch 10 → best half → 20), simulated

- Epochs 1000 of 2000, time saved 47.3%; of the 20 best by final fitness, 3 would not reach epoch 20

## Chosen rules (configs/search.yaml)

```yaml
calibration: runs\main_20261024\gen0_20ep\calibration.md
rules:
  not_learning: {epoch: 3}
  floor: null
  percentile: {10: 0.2}
  reach_elite: {check: [10, 15], q: 1.0}
  patience: {patience: 2, start: 15}
```
