# Short recipes vs 40 epochs (gen0_20ep vs runs\main_20261024\gen0_40ep)

| short recipe | networks | Spearman val F1 (95% CI) | Spearman fitness (95% CI) | Spearman fitness, best 30 (95% CI) | top 20% shared | hours (short / long) |
|---|---|---|---|---|---|---|
| 20 epochs | 100 | 0.974 (0.951–0.984) | 0.970 (0.946–0.981) | 0.707 (0.466–0.839) | 16 / 20 | 2.06 / 4.01 |

Inside the 40-epoch run (100 networks): best val F1 by epoch e vs the checkpoint val F1

| epoch | Spearman (95% CI) |
|---|---|
| 5 | 0.744 (0.626–0.826) |
| 10 | 0.824 (0.716–0.891) |
| 20 | 0.945 (0.904–0.969) |
| 30 | 0.991 (0.982–0.995) |
