# PCCV-Bench

A fully synthetic benchmark for **process-conditioned consistency verification**:
checking whether a set of learned detections is explained by an expected
production record, and measuring how well correction mechanisms (bounded
positional tolerance, temporal retry, record-side coverage) recover from
realistic kinds of mismatch.

**Contains zero real factory geometry.** Every scene, camera and detection is
generated from aggregate statistics. No real images, production records or
camera calibrations are included, and none are needed to run anything in the
quickstart. The six cameras are anonymised as `cam1` ... `cam6`.

The benchmark and the companion real-data study drive the same matcher through
the same harness (`pccv_bench/core/harness.py`), so a claim that a mechanism
behaves the same way on real and synthetic data is checked, not asserted.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
python -m pytest            # 55 tests, about 20 seconds
```

Python 3.10 or newer. Reference environment: Python 3.11, numpy 1.26, scipy 1.17, opencv 4.5.5. OpenCV must be between 4.5 and 4.13 (see below).

## Quickstart

```bash
python -m pccv_bench.cli generate --config pccv_bench/configs/medium_v1_2_fitted.yaml --seed 42 --out instances.json
python -m pccv_bench.cli ablate   --instances instances.json --out results.json --frozen-operating-point
python -m pccv_bench.cli report   --results results.json
```

`generate` produces 240 labelled instances per class for all six injection
classes (missing part, ghost part, in-bound and out-of-bound position shift,
snapshot desync, composition swap) plus the clean control: 1,680 instances.
`ablate` scores them through the full arm ladder: `flag_none` / `flag_all`
floors, four baselines, four box-only arms, and five two-way arms ending in the
frozen verifier `+all+twoway+offset_aware`. `--frozen-operating-point` scores
every arm, baselines included, at the companion study's frozen operating point
(box IoU 0.7, record coverage 0.4, 600 mm offset bound); use it for anything
compared with that study. `report` prints the summary and per-class tables.

## Verifying your install

The pinned seed-42 instance set is `instances/medium_v1_2_fitted_seed42.json`. It is
the canonical benchmark: all reference results in `results/` were computed on it. Its
content checksum is

```
8b2bee72cb8c01bda429156583b8aaac49956b5d5bb990e480ff3be0f76d3986
```

`tests/test_release_checksum.py` checks that checksum on the pinned file, and checks
that regenerating seed 42 reproduces the pinned set to within 1e-3 px / mm.

**Regeneration depends on OpenCV.** The synthetic cameras are fitted with OpenCV, so
the generated geometry is not bit-for-bit identical across builds:

- OpenCV 4.5.5, the reference build, reproduces the pinned set exactly (checksum above;
  it matched on two machines).
- OpenCV 4.6 to 4.13 agree with it to about 1e-4 px in one camera's fitted model. That
  is enough to change the checksum (geometry is rounded to six decimals before hashing)
  but not the instances in any meaningful way.
- **OpenCV 4.14 and 5.x are not supported.** They change the fitted model of `cam6` by
  about 0.2 px and generated coordinates by up to about 13 px, so regenerated instances and
  results differ from the pinned ones. `requirements.txt` pins `opencv-python-headless<4.14`,
  and the CLI warns if it finds a newer build.

If you need the exact published numbers, score the pinned instance set rather than
regenerating it.

Seed 42 is the release reference. Seeds 0-9 are the evaluation seeds and seed
1000 is reserved for tuning. Tiers: `easy_v1_2.yaml`, `medium_v1_2_fitted.yaml`,
`hard_v1_2.yaml` (fitted detector noise x0.5 / x1 / x1.5, occlusion 0.05 / 0.10
/ 0.20).

## What it reproduces from the real-data study, and what it does not

A pre-registered agreement study (`docs/PREREGISTRATION.md`, results in
`results/agreement_v2/`) compared the benchmark with the companion real-data
injection study at the same operating point:

- **Rank agreement.** Over the 13 shared arms, real and synthetic pooled
  detection rank the arms alike (Spearman rho 0.986, "supported" by the
  pre-registered rule); see `e4_rank_agreement.md` for every metric, its
  interval, and its category. Several per-class rankings agree only weakly.
- **Matched budget.** The real study's matched-budget pattern recurs: at no
  higher clean flag rate, the frozen verifier leads the symmetric and coherent
  baselines on missing-part recall and pooled detection, and the coherent
  baseline keeps the lower pooled false-error (`e3_matched_budget.md`).
- **Effect directions.** Of 15 pre-registered paired effects, 9 agree, 5 are
  inconclusive, and 1 disagrees (`e6_effect_directions.md`). The inconclusive
  ones involve record-side tolerance (spatial correction and temporal retry in
  the two-way arms), where the synthetic record-side check saturates on
  in-bound shifts and desynchronizations.
- **No numerical transfer.** The frozen verifier's clean flag rate is 35.42%
  at seed 42, against 5.83% presumed-clean disagreement in the annotation-fed
  real study. An exploratory noise ablation (`results/exploratory/`) puts the
  gap in the detector-noise model: without detector noise the rate is 1.25%.

Use the benchmark to compare methods and mechanisms, not to predict plant alarm
rates.

## What is in this repository

| Path | Contents |
|---|---|
| `pccv_bench/` | The generator, matcher, fault injectors, scoring and CLI. |
| `pccv_bench/core/` | Geometry, camera model, box-side matcher, two-way matcher, injectors, oracle, baselines, harness, and statistics. |
| `pccv_bench/configs/` | The three release tiers. |
| `instances/` | The pinned seed-42 instance set (medium tier) and its manifest. |
| `results/` | Reference results for all three tiers and the agreement-study outputs. |
| `docs/` | Pre-registration, noise-model description, datasheet. |

### Package layout

| Module | Purpose |
|---|---|
| `scene.py` | World-space synthetic scene generation; per-camera box-count distributions measured from real coarse labels. |
| `cameras.py` | Synthesizes ground-truth correspondences and fits the camera model on them, which exercises the fitting code with zero real geometry. |
| `observe.py` | Truth boxes to noisy detections (miss, false positive, jitter, split, merge). |
| `noise.py` | Detection-noise parameters and versioning. |
| `fit_noise.py` | Derives the miss and empty-image false-positive rates from a frozen detector evaluation and records provenance. |
| `synthetic_provider.py` | `SyntheticProvider`, which produces instances through the shared injectors. |
| `instance_io.py` | JSON (de)serialization for the `generate` to `ablate` split. |
| `ablate.py` | The arm ladder, re-exported from `core/harness.py`. |
| `cli.py` | `generate`, `ablate`, and `report` subcommands. |
| `agreement_stats.py`, `run_e*.py`, `run_seed_sweep.py` | The pre-registered agreement study (E1 to E7). |
| `release_validation.py` | Real-versus-synthetic rank correlation across a shared arm set. |

### Agreement-study scripts

`run_e4_rank_agreement.py`, `run_e6_effect_directions.py` and
`run_e7_alignment.py` compare against outputs of the companion real-data study
(`--real-verdicts`, `--real-summary`, `--real-matched-budget`). Those inputs
contain real per-frame results and are **not** released, so these three scripts
cannot be re-run from this repository. Their outputs are in
`results/agreement_v2/`. `run_e1_tiers.py`, `run_e2_seed_variance.py` and
`run_e3_matched_budget.py` need only this repository.

### Camera identifiers

The cameras are `cam1` ... `cam6`. They were renamed from the original
installation identifiers before release. Result files in `results/agreement_v2/`
that depend on the real-data study were produced before the rename and had the
identifiers replaced textually; the numbers are unaffected. Checksums stored
inside those files refer to the pre-rename instance files.

## Versioning

Every result file's `generation_meta` records `noise_version`, the config hash,
the seed and the git revision, so a reported number is traceable to exactly the
config and code that produced it. `v1.2-fitted` is the release version: it
fits jitter, split, merge, and false-positive size per camera from the measured
error statistics of one frozen detector on a chronological test split, with a
pooled fallback where a camera has too few samples. `miss_rate` and `fp_rate`
stay pooled. See `docs/NOISE_MODEL.md`. `score_mean` and `score_sigma` are
hand-set, since nothing downstream consumes a synthetic detection's score.

Git revisions stored inside result and instance files (`generation_meta.git_revision`)
refer to working commits made while the release was prepared; the published history is
a single squashed commit, so those hashes will not resolve here. The config hash, seed
and instance checksum are what identify a run.

## Limitations

- **One installation.** The generator is parameterised by aggregate statistics
  of one six-camera installation, so results describe that setting.
- **No numerical transfer** of alarm rates, as described above.
- **`composition_swap` recall is weaker than on real data**, because most
  synthetic groups hold a single part.
- **The noise model is fitted to one detector**, so conclusions are conditional
  on that detector's error profile.

## License

Apache-2.0, see `LICENSE`.

## Citation

See `CITATION.cff`. The archive DOI will be added once minted.
