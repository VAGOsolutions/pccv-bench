# Agreement-study outputs

Outputs of the pre-registered agreement study (`docs/PREREGISTRATION.md`).

| File | Produced from | Re-runnable from this repository |
|---|---|---|
| `e1_tiers.*` | The three seed-42 tier results in `results/` | Yes (`run_e1_tiers.py`) |
| `e2_seed_variance.*` | Seeds 0-9 of the medium tier | Yes (`run_seed_sweep.py`, `run_e2_seed_variance.py`) |
| `e3_matched_budget.*` | Seeds 0-9 plus tuning seed 1000 | Yes (`run_seed_sweep.py`, `run_e3_matched_budget.py`) |
| `e4_rank_agreement.*` | Medium tier plus the real study's per-instance verdicts | No: needs real-study outputs (not released) |
| `e6_effect_directions.*` | E3 plus the real study's summary and matched-budget file | No: needs real-study outputs (not released) |
| `e7_operating_point_alignment.*` | Before/after results plus the real study's summary | No: needs real-study outputs (not released) |

`e1_tiers.*` was regenerated from this repository. The other files were produced
before the cameras were renamed to `cam1` ... `cam6`. The identifiers inside them were replaced textually and absolute local paths were
reduced to file names. No number was changed. Config hashes, instance checksums
and git revisions stored inside them refer to the original development
repository and to the pre-rename instance files, so they will not match this
repository's.

The seed-42 tier results in `results/` and `results/exploratory/` were regenerated
from this repository after the rename and match the originals exactly in every
verdict-derived number. The instance checksum recorded for the medium tier
(`8b2bee72...`) is the checksum of the renamed pinned set.
