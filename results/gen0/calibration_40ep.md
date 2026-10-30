# Stop rule calibration on 100 fully trained candidates (runs\main_20261024\gen0_40ep)

B's elite bar (K = 20-th best fitness of the others) ~ 83.20; protected: the best 30 (bar ~ 82.20, 1.00 points below the elite bar); fitness 10 / 50 / 90%: 67.8 / 78.8 / 83.9

Chosen setting per rule: the one saving the most with no wrong cut (ties: the tighter).

## Fixed rules

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| not learning | 1 | 1 | 0.9% | 1.6% | 0 | -0.1 |
| patience | 0 | 15 | 1.4% | 1.4% | 0 | nan |

## D floor

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| epoch 10 < 0.40 | 4 | 4 | 3.0% | 4.0% | 0 | 81.8 |
| epoch 10 < 0.45 | 6 | 6 | 4.5% | 5.5% | 0 | 81.8 |
| epoch 10 < 0.50 | 10 | 10 | 7.5% | 7.9% | 0 | 81.8 |
| epoch 10 < 0.55 | 21 | 21 | 15.8% | 16.3% | 1 | 82.8 |
| epoch 10 < 0.60 | 35 | 35 | 26.2% | 24.4% | 2 | 82.8 |
| epoch 10 < 0.65 | 40 | 40 | 30.0% | 26.5% | 2 | 82.8 |
| epoch 10 < 0.70 | 61 | 61 | 45.8% | 41.7% | 4 | 83.2 |

Chosen: **epoch 10 < 0.50**

## A percentile

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| 10% at 20 | 11 | 11 | 5.5% | 4.4% | 0 | 74.1 |
| 20% at 20 | 21 | 21 | 10.5% | 8.3% | 0 | 77.1 |
| 30% at 20 | 30 | 30 | 15.0% | 11.7% | 1 | 82.7 |
| 50% at 20 | 50 | 50 | 25.0% | 20.9% | 2 | 82.8 |
| 5% / 10% at 10 / 20 | 12 | 12 | 7.5% | 7.5% | 0 | 81.8 |
| 10% / 20% at 10 / 20 | 23 | 23 | 14.2% | 12.8% | 1 | 82.8 |
| 20% / 30% at 10 / 20 | 33 | 33 | 21.5% | 18.6% | 2 | 82.8 |
| 30% / 50% at 10 / 20 | 51 | 51 | 33.0% | 29.5% | 2 | 82.8 |
| 40% / 60% at 10 / 20 | 60 | 60 | 40.0% | 35.4% | 2 | 82.8 |
| 5% / 10% / 15% at 5 / 10 / 20 | 18 | 18 | 12.8% | 13.0% | 1 | 82.8 |
| 10% / 15% / 25% at 5 / 10 / 20 | 31 | 31 | 21.4% | 20.4% | 1 | 82.8 |
| 20% / 30% / 50% at 5 / 10 / 20 | 51 | 51 | 35.5% | 32.1% | 2 | 82.8 |
| 30% / 40% / 60% at 5 / 10 / 20 | 61 | 61 | 44.8% | 41.0% | 3 | 83.2 |
| 40% / 50% / 70% at 5 / 10 / 20 | 70 | 70 | 52.8% | 47.5% | 6 | 83.2 |

Chosen: **20% at 20**

## B reach_elite

| setting | cut | stopped | epochs saved | time saved | wrong cuts (best 30 at the end) | best fitness cut |
|---|---|---|---|---|---|---|
| q 1.0 at 30 | 39 | 39 | 9.8% | 7.4% | 0 | 77.6 |
| q 0.95 at 30 | 65 | 65 | 16.2% | 14.4% | 2 | 83.1 |
| q 0.9 at 30 | 71 | 71 | 17.8% | 15.8% | 3 | 83.1 |
| q 0.8 at 30 | 75 | 75 | 18.8% | 16.9% | 5 | 83.1 |
| q 1.0 at 20/30 | 39 | 39 | 11.0% | 8.4% | 0 | 77.6 |
| q 0.95 at 20/30 | 66 | 66 | 26.5% | 22.3% | 3 | 83.1 |
| q 0.9 at 20/30 | 71 | 71 | 32.2% | 29.2% | 3 | 83.1 |
| q 0.8 at 20/30 | 76 | 76 | 36.0% | 32.4% | 6 | 83.1 |
| q 1.0 at 10/20/30 | 40 | 40 | 12.5% | 11.0% | 0 | 81.8 |
| q 0.95 at 10/20/30 | 66 | 66 | 34.5% | 31.3% | 3 | 83.1 |
| q 0.9 at 10/20/30 | 71 | 71 | 40.8% | 37.3% | 3 | 83.1 |
| q 0.8 at 10/20/30 | 76 | 76 | 49.2% | 44.4% | 6 | 83.1 |
| q 1.0 at 5/10/20/30 | 40 | 40 | 15.5% | 14.0% | 0 | 81.8 |
| q 0.95 at 5/10/20/30 | 66 | 66 | 36.2% | 33.4% | 3 | 83.1 |
| q 0.9 at 5/10/20/30 | 71 | 71 | 44.0% | 40.7% | 3 | 83.1 |
| q 0.8 at 5/10/20/30 | 78 | 78 | 55.8% | 51.3% | 8 | 83.7 |

Chosen: **q 1.0 at 5/10/20/30**

## Vetoes: protected candidates the settings cut

A rule whose every setting cuts one of these is off; decide knowingly whether that network is worth it.

| network | fitness (rank) | best val F1 by epoch 5 / 10 / 20 → end | cut by | architecture |
|---|---|---|---|---|
| 076_fbe32e4cbb0a0bda | 83.72 (16) | 56.6 / 72.4 / 79.9 → 84.0 | B 1 of 16 (loosest: q 0.8 at 5/10/20/30) | s64 mbconv/concat d2 w256 k3 none | s16 double_conv/concat d2 w48 k7 bn |
| 085_1893bb86a78c9049 | 83.20 (21) | 51.4 / 68.9 / 80.0 → 83.7 | D 1 of 7 (loosest: epoch 10 < 0.70); A 2 of 14 (loosest: 30% / 40% / 60% at 5 / 10 / 20); B 1 of 16 (loosest: q 0.8 at 5/10/20/30) | s64 conv/add d2 w32 k1 ln | s32 dw_sep/add d5 w256 k7 bn | s4 bottleneck/add d5 w64 k3 none |
| 074_1adaa162f4e6c832 | 83.09 (23) | 59.8 / 68.6 / 77.0 → 83.9 | D 1 of 7 (loosest: epoch 10 < 0.70); A 1 of 14 (loosest: 40% / 50% / 70% at 5 / 10 / 20); B 12 of 16 (loosest: q 0.95 at 30) | s32 dw_sep/concat d4 w384 k3 ln | s16 convnext/none d4 w96 k1 bn | s4 conv/add d5 w128 k5 ln |
| 070_51334fcc56213f08 | 82.94 (24) | 63.4 / 73.5 / 76.9 → 83.0 | A 1 of 14 (loosest: 40% / 50% / 70% at 5 / 10 / 20); B 3 of 16 (loosest: q 0.8 at 20/30) | s32 double_conv/concat d4 w96 k5 ln | s16 bottleneck/concat d3 w192 k3 bn | s4 conv/add d4 w128 k3 none |
| 042_932878fabd40b9a7 | 82.84 (25) | 50.1 / 50.1 / 72.0 → 82.5 | D 4 of 7 (loosest: epoch 10 < 0.55); A 10 of 14 (loosest: 50% at 20); B 12 of 16 (loosest: q 0.95 at 30) | s32 bottleneck/add d1 w192 k3 ln | s16 dw_sep/concat d3 w48 k1 ln | s8 conv/add d6 w48 k3 bn |
| 092_1c3a6620672b60bc | 82.75 (28) | 47.8 / 55.1 / 67.3 → 82.3 | D 3 of 7 (loosest: epoch 10 < 0.60); A 8 of 14 (loosest: 30% at 20); B 11 of 16 (loosest: q 0.9 at 30) | s16 mbconv/concat d5 w48 k3 bn | s8 mbconv/add d4 w64 k7 bn |
| 027_559cc8fdfee71fbf | 82.62 (29) | 56.1 / 71.3 / 76.3 → 81.6 | A 1 of 14 (loosest: 40% / 50% / 70% at 5 / 10 / 20); B 4 of 16 (loosest: q 0.8 at 30) | s64 convnext/none d3 w24 k3 ln | s4 dw_sep/add d5 w96 k7 bn |
| 036_ff91753454fdecaa | 82.20 (30) | 65.5 / 73.5 / 78.0 → 81.6 | B 4 of 16 (loosest: q 0.8 at 30) | s64 double_conv/add d6 w32 k1 none | s16 convnext/add d2 w64 k7 bn | s8 bottleneck/add d6 w48 k7 none | s4 double_conv/concat d4 w192 k1 ln |

## All chosen rules together

- Stopped 55 / 100 ({'floor': 4, 'percentile': 11, 'reach_elite': 24, 'patience': 15, 'not learning': 1}); time saved **18.1%**, wrong cuts 0 (B saves more in the GA: its bar rises)

## C: successive halving (epoch 10 → best half → epoch 20 → best half → 40), simulated

- Epochs 2000 of 4000, time saved 45.1%; of the 20 best by final fitness, 2 would not reach epoch 40

## Chosen rules (configs/search.yaml)

```yaml
calibration: runs\main_20261024\gen0_40ep\calibration.md
rules:
  not_learning: {epoch: 3}
  floor: {10: 0.5}
  percentile: {20: 0.2}
  reach_elite: {check: [5, 10, 20, 30], q: 1.0}
  patience: {patience: 5, start: 30}
```
