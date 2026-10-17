# E7 — operating-point alignment: old vs new

- git revision: `fe748e8b677613ccd0b9298e7501e8ab145bac69-dirty`
- real reference: `injection_summary.json` (git_revision `42b2d28cf02d7ee2bf5052cb17a6756f05187230`, 5903 instances)
- old (misaligned) run git revision: `6ec378f959a5fccb4213769d2707d6f85da85877`
- new (aligned) run git revision: `fe748e8b677613ccd0b9298e7501e8ab145bac69`, frozen point: {'metric': 'iou', 'threshold': 0.7, 'correction_offset_mm': 600.0, 'use_temporal': True, 'twoway_threshold': 0.4}

## 5-arm ladder Spearman rho (geometry, +offset, +temporal, +offset+temporal, +all+twoway)

| | rho |
|---|---|
| old (misaligned) | 0.9747 |
| new (aligned) | 0.9487 |

## Pooled detection rate, per arm

| arm | real | synthetic (old, misaligned) | synthetic (new, aligned) |
|---|---|---|---|
| `geometry` | 0.5047 | 0.3906 | 0.7115 |
| `+offset` | 0.4429 | 0.3771 | 0.6948 |
| `+temporal` | 0.4841 | 0.3896 | 0.7115 |
| `+offset+temporal` | 0.4381 | 0.3771 | 0.6948 |
| `+all+twoway` | 0.9083 | 0.8667 | 0.9719 |

## Clean false-error rate (presumed-clean disagreement), per arm

| arm | real | synthetic (old, misaligned) | synthetic (new, aligned) |
|---|---|---|---|
| `+all+twoway` | 0.0690 | 0.3292 | 0.3583 |
| `+all+twoway+offset_aware` | 0.0583 | 0.3167 | 0.3542 |
