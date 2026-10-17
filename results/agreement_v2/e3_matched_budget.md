# E3 — synthetic matched-budget protocol

- git revision: `ac95907ae89d7a86570e30c606ab830c3ea8c304`
- tuning seed: 1000; evaluation seeds: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
- budget: **0.3708** (frozen verifier's (+all+twoway+offset_aware) own tuning clean flag rate)
- selected thresholds: symmetric=0.2, coherent=0.2

## Per-seed and mean (verifier / symmetric / coherent)

| metric | method | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | seed 5 | seed 6 | seed 7 | seed 8 | seed 9 | mean | range |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `clean_flag_rate` | verifier | 0.3000 | 0.3250 | 0.2917 | 0.3000 | 0.2917 | 0.3167 | 0.3250 | 0.3208 | 0.3000 | 0.3208 | 0.3092 | [0.2917, 0.3250] |
| `clean_flag_rate` | symmetric | 0.2625 | 0.3167 | 0.2875 | 0.2583 | 0.2833 | 0.3000 | 0.3042 | 0.2917 | 0.2667 | 0.3083 | 0.2879 | [0.2583, 0.3167] |
| `clean_flag_rate` | coherent | 0.2583 | 0.3000 | 0.2833 | 0.2500 | 0.2833 | 0.2833 | 0.2917 | 0.2750 | 0.2583 | 0.3042 | 0.2787 | [0.2500, 0.3042] |
| `missing_part_recall` | verifier | 0.9833 | 0.9750 | 0.9792 | 0.9750 | 0.9875 | 0.9833 | 0.9583 | 0.9833 | 0.9833 | 0.9542 | 0.9762 | [0.9542, 0.9875] |
| `missing_part_recall` | symmetric | 0.8250 | 0.8500 | 0.8000 | 0.8375 | 0.8208 | 0.8250 | 0.7708 | 0.7833 | 0.7708 | 0.7792 | 0.8063 | [0.7708, 0.8500] |
| `missing_part_recall` | coherent | 0.8208 | 0.8458 | 0.7875 | 0.8250 | 0.8125 | 0.8042 | 0.7583 | 0.7708 | 0.7583 | 0.7708 | 0.7954 | [0.7583, 0.8458] |
| `ghost_part_recall` | verifier | 0.9792 | 0.9792 | 0.9792 | 0.9750 | 0.9625 | 0.9625 | 0.9792 | 0.9708 | 0.9667 | 0.9792 | 0.9733 | [0.9625, 0.9792] |
| `ghost_part_recall` | symmetric | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | [1.0000, 1.0000] |
| `ghost_part_recall` | coherent | 1.0000 | 1.0000 | 1.0000 | 0.9958 | 0.9958 | 0.9958 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9988 | [0.9958, 1.0000] |
| `pooled_detection` | verifier | 0.9677 | 0.9708 | 0.9677 | 0.9594 | 0.9667 | 0.9615 | 0.9646 | 0.9677 | 0.9688 | 0.9615 | 0.9656 | [0.9594, 0.9708] |
| `pooled_detection` | symmetric | 0.8875 | 0.9177 | 0.8938 | 0.8823 | 0.8969 | 0.8917 | 0.8865 | 0.8865 | 0.8865 | 0.8906 | 0.8920 | [0.8823, 0.9177] |
| `pooled_detection` | coherent | 0.8812 | 0.9062 | 0.8844 | 0.8708 | 0.8885 | 0.8802 | 0.8760 | 0.8760 | 0.8771 | 0.8802 | 0.8821 | [0.8708, 0.9062] |
| `in_bound_flag_rate` | verifier | 0.8125 | 0.7500 | 0.7167 | 0.8333 | 0.7125 | 0.7875 | 0.7500 | 0.7708 | 0.7458 | 0.7708 | 0.7650 | [0.7125, 0.8333] |
| `in_bound_flag_rate` | symmetric | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.9958 | 1.0000 | 1.0000 | 0.9996 | [0.9958, 1.0000] |
| `in_bound_flag_rate` | coherent | 0.7958 | 0.7208 | 0.6625 | 0.7792 | 0.6917 | 0.7250 | 0.7083 | 0.7375 | 0.7333 | 0.7458 | 0.7300 | [0.6625, 0.7958] |
| `pooled_false_error` | verifier | 0.5563 | 0.5375 | 0.5042 | 0.5667 | 0.5021 | 0.5521 | 0.5375 | 0.5458 | 0.5229 | 0.5458 | 0.5371 | [0.5021, 0.5667] |
| `pooled_false_error` | symmetric | 0.6312 | 0.6583 | 0.6438 | 0.6292 | 0.6417 | 0.6500 | 0.6521 | 0.6438 | 0.6333 | 0.6542 | 0.6438 | [0.6292, 0.6583] |
| `pooled_false_error` | coherent | 0.5271 | 0.5104 | 0.4729 | 0.5146 | 0.4875 | 0.5042 | 0.5000 | 0.5062 | 0.4958 | 0.5250 | 0.5044 | [0.4729, 0.5271] |

## Paired differences (verifier minus baseline), pooled across seeds, instance-bootstrap 95% CI

| baseline | metric | difference | 95% CI | significant |
|---|---|---|---|---|
| symmetric | `clean_flag_rate` | 0.0212 | [0.0087, 0.0338] | **yes** |
| symmetric | `missing_part_recall` | 0.1700 | [0.1550, 0.1867] | **yes** |
| symmetric | `ghost_part_recall` | -0.0267 | [-0.0333, -0.0204] | **yes** |
| symmetric | `pooled_detection` | 0.0736 | [0.0675, 0.0800] | **yes** |
| symmetric | `in_bound_flag_rate` | -0.2346 | [-0.2521, -0.2175] | **yes** |
| symmetric | `pooled_false_error` | -0.1067 | [-0.1171, -0.0956] | **yes** |
| coherent | `clean_flag_rate` | 0.0304 | [0.0175, 0.0433] | **yes** |
| coherent | `missing_part_recall` | 0.1808 | [0.1654, 0.1975] | **yes** |
| coherent | `ghost_part_recall` | -0.0254 | [-0.0321, -0.0192] | **yes** |
| coherent | `pooled_detection` | 0.0835 | [0.0772, 0.0898] | **yes** |
| coherent | `in_bound_flag_rate` | 0.0350 | [0.0225, 0.0479] | **yes** |
| coherent | `pooled_false_error` | 0.0327 | [0.0231, 0.0415] | **yes** |
