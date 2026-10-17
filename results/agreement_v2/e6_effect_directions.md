# E6 — effect-direction agreement

- git revision: `86ad56108acab74d4a86e6cd45bf5c1171ae3619`
- headline: **9 agree / 5 inconclusive / 1 disagree** out of 15

| ID | claim | contrast | metric | expected sign | real | synthetic diff | synthetic 95% CI | outcome |
|---|---|---|---|---|---|---|---|---|
| C1 | The record side catches added parts | `+all+twoway - +offset+temporal` | `class:ghost_part` | + | 1.0000 - 0.0397 = +0.9603 | +0.8458 | [0.8000, 0.8917] | **agree** |
| C2 | A second alarm channel raises clean disagreement | `+all+twoway - +offset+temporal` | `clean_flag_rate` | + | 0.0690 - 0.0402 = +0.0288 | +0.1833 | [0.1375, 0.2333] | **agree** |
| C3 | Spatial correction lowers clean disagreement | `+offset+twoway - geometry+twoway` | `clean_flag_rate` | - | 0.0690 - 0.1152 = -0.0462 | -0.0292 | [-0.0542, -0.0083] | **agree** |
| C4 | Spatial correction lowers in-bound flags | `+offset+twoway - geometry+twoway` | `class:position_shift_x_inbound` | - | 0.7155 - 0.9378 = -0.2223 | +0.0000 | [-0.0153, 0.0153] | **inconclusive** |
| C5 | Spatial correction masks missing parts | `+offset+twoway - geometry+twoway` | `class:missing_part` | - | 0.7508 - 0.8477 = -0.0969 | -0.0125 | [-0.0292, 0.0000] | **inconclusive** |
| C6 | Temporal retry recovers desync | `+temporal+twoway - geometry+twoway` | `class:snapshot_desync` | - | 0.6759 - 0.9532 = -0.2773 | +0.0000 | [-0.0153, 0.0153] | **inconclusive** |
| C7 | The offset-aware record side lowers in-bound flags | `+all+twoway+offset_aware - +all+twoway` | `class:position_shift_x_inbound` | - | 0.5304 - 0.7113 = -0.1809 | -0.2083 | [-0.2583, -0.1583] | **agree** |
| C8 | ...and lowers clean disagreement | `+all+twoway+offset_aware - +all+twoway` | `clean_flag_rate` | - | 0.0583 - 0.0690 = -0.0107 | -0.0042 | [-0.0125, 0.0000] | **inconclusive** |
| C9 | ...at some missing-part cost | `+all+twoway+offset_aware - +all+twoway` | `class:missing_part` | - | 0.7308 - 0.7415 = -0.0107 | -0.0042 | [-0.0125, 0.0000] | **inconclusive** |
| C10 | The box side alone rarely sees added parts | `+offset+temporal vs level 0.1` | `class:ghost_part` | below 0.1 | 0.0397 | +0.1542 | [0.1083, 0.2000] | **disagree** |
| C11 | At native threshold, symmetric alarms far more on clean data | `baseline_symmetric - +all+twoway+offset_aware` | `clean_flag_rate` | + | 0.4200 - 0.0583 = +0.3617 | +0.1833 | [0.1333, 0.2375] | **agree** |
| C12 | ...and so does coherent | `baseline_coherent - +all+twoway+offset_aware` | `clean_flag_rate` | + | 0.4012 - 0.0583 = +0.3429 | +0.1708 | [0.1250, 0.2208] | **agree** |
| C13 | At a matched budget the verifier leads on missing parts (from E3) | `verifier vs ['symmetric', 'coherent'] (E3)` | `missing_part_recall` | + | symmetric: +0.1938; coherent: +0.2662 (the real study's matched-budget comparison, Table 2 of the companion manuscript) | symmetric: +0.1700 [0.155, 0.1867]; coherent: +0.1808 [0.1654, 0.1975] | — | **agree** |
| C14 | At a matched budget the verifier leads on pooled detection (from E3) | `verifier vs ['symmetric', 'coherent'] (E3)` | `pooled_detection` | + | symmetric: +0.1187; coherent: +0.1962 (the real study's matched-budget comparison, Table 2 of the companion manuscript) | symmetric: +0.0736 [0.0675, 0.08]; coherent: +0.0835 [0.0772, 0.0898] | — | **agree** |
| C15 | At a matched budget coherent keeps the lower pooled false-error (from E3) | `verifier vs ['coherent'] (E3)` | `pooled_false_error` | - | coherent: -0.0713 (the real study's matched-budget comparison, Table 2 of the companion manuscript) | coherent: -0.0327 [-0.0415, -0.0231] | — | **agree** |
