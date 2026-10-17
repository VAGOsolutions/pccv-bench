# PCCV-Bench agreement study — pre-registration

> **Note on revisions.** Commit and revision identifiers in this document refer to the original
> development repository, whose history is not part of this release.
>
> **Status:** frozen once committed. Commit this file **before** any E1–E6 run. Every result is
> reported against it, including the ones that disagree. Any later change goes in the amendment log
> at the bottom, with a date and a reason, and never replaces an entry.
>
> **Disclosure.** The authors have already seen the synthetic v1.2 results
> (`results/report_medium_v1_2_fitted.txt`). Those were produced at a
> *different* operating point from the real study (see §1), so they are treated as pre-study history,
> not as data for this study. The contrast list in §4 was built by a rule (every paired claim in the
> companion real-data manuscript that the shared arms can express), not by picking what v1.2 happened to reproduce.

---

## 1. Operating point (the reason for this study's reruns)

The real study scores every arm at its frozen config
(`pccv_bench/frozen_operating_point.json`; applied with `dataclasses.replace`):

| | Box side | Record side | Offset bound | Temporal |
|---|---|---|---|---|
| **Real study (frozen)** | IoU ≥ 0.7 | containment ≥ 0.4 | 600 mm | on |
| **PCCV-Bench v1.2 run** | containment ≥ 0.3 (harness default; `pccv_bench/cli.py` does not apply the frozen override) | ≥ 0.7 (`medium_v1_2_fitted.yaml`) | 600 mm | on |
| easy / medium / hard configs | containment ≥ 0.3 | ≥ 0.3 | 600 mm | on |

The existing ρ = 0.9747 and the 32.92% vs 6.90% comparison therefore compare **different operating
points on both matcher sides**.

**Primary analysis for this study:** every synthetic arm, baselines included, is scored with the real
frozen config substituted exactly as the real study does it (`metric="iou"`, `threshold=0.7`,
`twoway_threshold=0.4`). The v1.2 misaligned numbers are reported only as history.

---

## 2. Fixed inputs

- **Generator:** the PCCV-Bench release tag (`pccv-bench-v1.2` or its successor), from a clean tree. No `-dirty` revisions.
- **Noise:** `v1.2-fitted` (per-camera; see `docs/NOISE_MODEL.md`).
- **Real reference:** the real study's `injection_summary.json` (5,903 instances; not released). For bootstrap work use the per-instance `verdicts.json` from the same revision (not released).
- **Instances per class:** 240 (medium).
- **Seeds:** evaluation seeds 0–9. Seed 42 is the release reference and is reported separately. The tuning seed for E3 is 1000.
- **Statistics:** 95% intervals from instance-level bootstrap (10,000 replicates) on the synthetic side and capture-run cluster bootstrap on the real side, the same as the real study.

---

## 3. E4 — rank agreement

**Arm set (13, fixed):**

- Baselines: `baseline_envelope`, `baseline_symmetric`, `baseline_coherent`, `baseline_pairwise`.
- Box-only arms: `geometry`, `+offset`, `+temporal`, `+offset+temporal`.
- Two-way arms: `geometry+twoway`, `+offset+twoway`, `+temporal+twoway`, `+all+twoway`, `+all+twoway+offset_aware`.

`flag_none` and `flag_all` are excluded, because as floors they rank trivially.

**Metrics, each ranked separately across the 13 arms:**

1. Pooled detection rate (primary).
2. Clean-control flag rate (synthetic false-error vs real presumed-clean disagreement).
3. Per-class flag rate for each of the seven classes.

**Statistics:**

- Spearman ρ and Kendall τ-b.
- Permutation p-value from 100,000 random permutations of the synthetic ranks (13! is too large to enumerate).
- 95% CI for ρ from a joint bootstrap: synthetic instances and real capture runs resampled independently.
- The 5-arm ladder (geometry, +offset, +temporal, +offset+temporal, +all+twoway) reported alongside, for continuity with the earlier artefacts.

**Reading rule:**

- Rank agreement is **supported** for a metric if the ρ CI lower bound is > 0.5 and p < 0.05.
- It is **weak** if ρ > 0 but either condition fails.
- It is **absent** otherwise.

No threshold is tuned after the fact.

---

## 4. E6 — effect-direction agreement

**Rule for inclusion:** every paired claim in the companion real-data manuscript that can be written
as a difference between two shared arms, or between an arm and a fixed level, on one metric.

**Real values** below are taken from `injection_summary.json`. Differences are the first arm minus the second.

| ID | Claim (companion manuscript) | Contrast | Metric | Real | Expected sign |
|---|---|---|---|---|---|
| C1 | The record side catches added parts | `+all+twoway` − `+offset+temporal` | ghost-part flag | 1.000 − 0.040 | + |
| C2 | A second alarm channel raises clean disagreement | `+all+twoway` − `+offset+temporal` | clean flag | 0.069 − 0.040 | + |
| C3 | Spatial correction lowers clean disagreement | `+offset+twoway` − `geometry+twoway` | clean flag | 0.069 − 0.115 | − |
| C4 | Spatial correction lowers in-bound flags | `+offset+twoway` − `geometry+twoway` | in-bound flag | 0.716 − 0.938 | − |
| C5 | Spatial correction masks missing parts | `+offset+twoway` − `geometry+twoway` | missing-part flag | 0.751 − 0.848 | − |
| C6 | Temporal retry recovers desync | `+temporal+twoway` − `geometry+twoway` | desync flag | 0.676 − 0.953 | − |
| C7 | The offset-aware record side lowers in-bound flags | `+all+twoway+offset_aware` − `+all+twoway` | in-bound flag | 0.530 − 0.711 | − |
| C8 | …and lowers clean disagreement | same | clean flag | 0.058 − 0.069 | − |
| C9 | …at some missing-part cost | same | missing-part flag | 0.731 − 0.742 | − |
| C10 | The box side alone rarely sees added parts | `+offset+temporal` vs level 0.10 | ghost-part flag | 0.040 | below 0.10 |
| C11 | At native threshold, symmetric alarms far more on clean data | `baseline_symmetric` − `+all+twoway+offset_aware` | clean flag | 0.420 − 0.058 | + |
| C12 | …and so does coherent | `baseline_coherent` − `+all+twoway+offset_aware` | clean flag | 0.401 − 0.058 | + |
| C13 | At a matched budget the verifier leads on missing parts (from E3) | verifier − symmetric; verifier − coherent | missing-part flag | +19.38; +26.62 pts | + |
| C14 | At a matched budget the verifier leads on pooled detection (from E3) | verifier − symmetric; verifier − coherent | pooled detection | +11.87; +19.62 pts | + |
| C15 | At a matched budget coherent keeps the lower pooled false-error (from E3) | coherent − verifier | pooled false-error | 14.12 − 21.24 | − |

C13–C15 use the real matched-budget numbers from
the real study's matched-budget comparison (Table 2 of the companion manuscript).

**Outcome per contrast** (synthetic paired difference, 95% bootstrap CI):

- **agree:** the CI excludes 0 with the expected sign. For C10, the upper CI bound is < 0.10.
- **inconclusive:** the CI includes 0.
- **disagree:** the CI excludes 0 with the opposite sign.

**Headline:** the counts of agree / inconclusive / disagree out of 15, with every row shown. The
magnitude of a difference is reported but not scored, since no numerical transfer is claimed.

---

## 5. E3 — synthetic matched-budget protocol

This mirrors the real tuning-only protocol:

1. Generate a **tuning** set (seed 1000) and use the **evaluation** sets (seeds 0–9).
2. On tuning data, the budget is the frozen verifier's (`+all+twoway+offset_aware`, frozen config) clean flag rate.
3. For `baseline_symmetric` and `baseline_coherent`, choose the threshold from the real study's grid that maximizes pooled detection subject to tuning clean flag rate ≤ budget. Missing-part recall is never consulted.
4. Freeze, then evaluate on each evaluation seed. Report the mean and the across-seed range.

---

## 6. E1 / E2 — tiers and seed variance

- **Tiers** are defined relative to the fitted noise rather than the hand-set v0.1:
  - **easy:** jitter, miss, FP, and merge rates × 0.5;
  - **medium:** v1.2-fitted as is;
  - **hard:** × 1.5, with rates capped at 1.
  - Occlusion rate 0.05 / 0.10 / 0.20.
- **Tier expectation, checked not tuned:** for the frozen verifier, pooled detection is non-increasing and clean flag rate non-decreasing from easy to hard. If this fails, report it and fix the tier definition in a dated amendment.
- **Seed variance:** E2 reports, for every arm, the mean and range across seeds 0–9, plus how often the E4 primary ρ keeps its reading-rule category.

---

## 7. Amendment log

All amendments below were written **after** the E0–E7 results were seen (runs of 2026-09-24, final
revision `e9e5533` of the original repository). None of them changes a threshold, an arm, a metric, a seed, or a
contrast. Where an amendment completes an under-specified rule, the outcome is reported under both the
completed rule and the one the runner actually used.

| Date | Change | Reason |
|---|---|---|
| 2026-09-24 | **A1. E4 ranks are tie-aware inside the resampling loops.** Commit `17bc3c8` replaces position-broken tie ranks with average ranks in the permutation and bootstrap loops; E4 is rerun from a clean tree. | §3 specifies Spearman ρ, which averages ties by definition. The first run broke ties by position inside the loops, on the assumption that ties between resampled rates are measure-zero. They are not: arms that share a mechanism give identical rates in every resample (every two-way arm flags every ghost part; synthetic `geometry` and `+temporal` coincide). The loops therefore resampled a different statistic from the point estimate, so several CIs excluded their own point (pooled detection ρ 0.986 vs CI [0.907, 0.978]; ghost ρ 0.927 vs CI [0.962, 0.978]; 5-arm CI [0.8, 0.8]) and the 5-arm permutation p-values fell below the 1/120 floor that five arms allow. This is an implementation correction, not a change of statistic; the first-run numbers are kept in git history (`86ad561`) and are not reported.

The correction changes reported categories at the 13-arm level, not only interval widths: `class:missing_part` (weak, CI [0.341, 0.940] -> supported, CI [0.701, 0.968]), `class:position_shift_x_inbound` (weak, CI [0.418, 0.522] -> supported, CI [0.671, 0.739]), and `class:snapshot_desync` (weak, CI [0.269, 0.577] -> supported, CI [0.643, 0.872]) move from weak to supported; `class:position_shift_out` stays weak (rho 0.425 in both runs). The 5-arm ladder's `pooled_detection` moves from a spurious "supported" (p=0.00001, impossible for 5 items under 100,000 permutations of only 5!=120 orderings) to the honest "weak" (p=0.067); its `class:ghost_part` moves the same way. Seven of eight 13-arm metrics are supported after the correction; only `class:position_shift_out` remains weak. |
| 2026-09-24 | **A2. Compound contrasts C13 and C14.** Each compares the verifier with two baselines. Outcome: *agree* only if both sub-contrasts agree; *disagree* if either disagrees; otherwise *inconclusive*. | §4 did not say how a two-part row is scored. The runner chose this rule; it is the strictest of the natural options. Both sub-contrasts agree for C13 and C14, so every candidate rule gives the same outcome. |
| 2026-09-24 | **A3. C10 outcome rule completed.** *Disagree* if the lower 95% CI bound is ≥ 0.10 (the CI lies wholly on the wrong side of the level, the analogue of "CI excludes 0 with the opposite sign"); *inconclusive* if the CI contains 0.10. | §4 defined only *agree* (upper bound < 0.10) for this level contrast. The runner used "point estimate ≥ 0.10 → disagree". Synthetic CI [0.108, 0.200]: both rules give **disagree**. |
| 2026-09-24 | **A4. E1 tier expectation failed; tiers kept unchanged.** Pooled detection of the frozen verifier is 0.956 / 0.964 / 0.960 (easy / medium / hard), not non-increasing; the clean flag rate is non-decreasing (0.188 / 0.354 / 0.471). The failure is reported. The tier definitions are **not** refitted. | §6 says to "fix the tier definition in a dated amendment" if the check fails. Refitting the multipliers until detection falls monotonically would tune the benchmark to its own expectation after seeing the outcome, which this document exists to prevent. The expectation itself was mis-specified: detection counts every alarm on a faulted instance, and heavier detection noise both masks faults and creates alarms. The data are consistent with both effects: the four box-only arms fall monotonically (e.g. `geometry` 0.731 / 0.712 / 0.663) and no baseline rises (three fall, `baseline_envelope` stays at 0.977), while the five two-way arms stay within 0.8 points, consistent with record-side alarms from missed detections rising with noise and offsetting the masking. The tiers are therefore described by what they do order monotonically, the nuisance-alarm rate, and detection is reported as it came out. |
| 2026-09-24 | **A5. E2 and E3 intermediates regenerated from a clean tree on a second machine.** Seeds 0–9 regenerated and re-ablated at `17bc3c8` (clean); E2, E3, and E6 rerun from them. | The first E2 intermediates were produced while untracked scripts sat in the working tree (`-dirty`). The scoring code was unchanged, so the rerun is a provenance fix and a cross-machine determinism check (see the rerun notes in `results/agreement_v2/`). |
| 2026-09-24 | **A6. E7 comparison file regenerated at a clean revision** (`17bc3c8`); content identical. | The committed `e7_operating_point_alignment.md` was stamped `fe748e8…-dirty`. |
