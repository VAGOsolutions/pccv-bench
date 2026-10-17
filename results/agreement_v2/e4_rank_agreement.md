# E4 — rank agreement

- git revision: `17bc3c8f8e46eb33cda38b1dd2891ee5d5597fe1`
- 13 arms: ['baseline_envelope', 'baseline_symmetric', 'baseline_coherent', 'baseline_pairwise', 'geometry', '+offset', '+temporal', '+offset+temporal', 'geometry+twoway', '+offset+twoway', '+temporal+twoway', '+all+twoway', '+all+twoway+offset_aware']

## 13-arm

| metric | rho | tau-b | perm p (100k) | rho 95% CI (joint boot, 10k) | category |
|---|---|---|---|---|---|
| `pooled_detection` | 0.9862 | 0.9477 | 0.00001 | [0.9295, 0.9945] | **supported** |
| `clean_flag_rate` | 0.9103 | 0.8105 | 0.00004 | [0.8855, 0.9545] | **supported** |
| `class:missing_part` | 0.9386 | 0.8438 | 0.00004 | [0.7011, 0.9679] | **supported** |
| `class:ghost_part` | 0.9266 | 0.8664 | 0.00009 | [0.8597, 0.9392] | **supported** |
| `class:position_shift_x_inbound` | 0.7371 | 0.6565 | 0.00591 | [0.6712, 0.7391] | **supported** |
| `class:position_shift_out` | 0.4246 | 0.3529 | 0.15330 | [0.3973, 0.8107] | **weak** |
| `class:snapshot_desync` | 0.7737 | 0.6725 | 0.00323 | [0.6430, 0.8720] | **supported** |
| `class:composition_swap` | 0.9319 | 0.8244 | 0.00001 | [0.8245, 0.9650] | **supported** |

## 5-arm-ladder

| metric | rho | tau-b | perm p (100k) | rho 95% CI (joint boot, 10k) | category |
|---|---|---|---|---|---|
| `pooled_detection` | 0.9487 | 0.8944 | 0.06731 | [0.9487, 0.9487] | **weak** |
| `clean_flag_rate` | 0.7000 | 0.6000 | 0.23479 | [0.6325, 0.9000] | **weak** |
| `class:missing_part` | 0.9487 | 0.8944 | 0.06739 | [0.0000, 0.9487] | **weak** |
| `class:ghost_part` | 0.9487 | 0.8944 | 0.06731 | [0.9487, 0.9733] | **weak** |
| `class:position_shift_x_inbound` | 0.6325 | 0.4472 | 0.26898 | [0.6325, 0.6325] | **weak** |
| `class:position_shift_out` | 0.0000 | 0.0000 | 1.00000 | [0.0000, 0.7454] | **absent** |
| `class:snapshot_desync` | 0.9000 | 0.8000 | 0.08346 | [0.7906, 1.0000] | **weak** |
| `class:composition_swap` | 0.7906 | 0.6708 | 0.13565 | [0.6325, 0.9487] | **weak** |

## Seed 0-9 stability (primary metric: pooled detection)

- seed 42 rho: 0.9862
- per-seed rho (0-9): [0.96, 0.9821, 0.9531, 0.9779, 0.9545, 0.9601, 0.96, 0.9835, 0.9503, 0.9615]
- reading-rule category stable across seeds: **True**
