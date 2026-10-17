# Datasheet: PCCV-Bench v1.2

## Motivation

Checking whether a camera view agrees with a per-instance production record is
a recurring task in industrial monitoring, but the evaluation data behind such
checks are almost never released. PCCV-Bench provides an executable, fully
synthetic benchmark so that verification methods can be compared, and their
tolerance mechanisms stress-tested, without access to any plant data.

## Composition

Each instance is one camera view with an expected record and a set of detections.
The expected record is a list of world-space rectangles (position, size,
rotation). The detections are pixel-space boxes with scores. Each instance
carries ground truth in two parts: the fault type (what is wrong with the pair)
and the expected action (alarm, pass, or recover).

Seven classes, 240 instances each in the pinned medium set (1,680 in total):
clean control, missing part, ghost part, in-bound position shift, out-of-bound
position shift, snapshot desynchronization, and composition swap. Six cameras,
`cam1` to `cam6`. Every part is an instance of one class; no instance carries a
type, category, or product attribute.

## Collection and generation

Nothing is collected. Every scene, camera and detection is generated from a
seeded random process (`pccv_bench/scene.py`, `cameras.py`, `observe.py`,
`synthetic_provider.py`). Faults are injected by perturbing the record, not the
image, so ground truth is exact. A matcher-independent geometric oracle decides
whether each injected fault is observable at all.

## What aggregate real statistics went in

The generator is parameterised by aggregate statistics of one six-camera
installation. These are the only real-world inputs:

- the per-camera distribution of the number of detected groups per image,
  measured from 3,000 real coarse labels (`scene.py`);
- example part pitch, size ranges and an axis-aligned fraction (`scene.py`);
- per-camera target camera-calibration reprojection error, 1.10 to 8.54 px
  (`cameras.py`), and the camera-model type used per camera;
- a fixed world-space x offset of -450.5 mm applied to one table's parts before
  projection (`core/matching.py`);
- the per-camera detector-noise parameters, fitted from one frozen detector's
  predictions on a chronological test split (`configs/`, `docs/NOISE_MODEL.md`).

No real image, record, geometry or calibration is included. The camera models
are fitted on synthetic correspondences, and the synthetic camera pose is
tuned, not measured.

## Intended uses

Comparing verification methods and correction mechanisms (positional
tolerance, temporal retry, record-side coverage) at matched nuisance-alarm
budgets; testing that a matcher behaves as specified on cases with known
answers.

## Non-uses

It is **not a numerical proxy** for plant alarm rates. The frozen verifier's
clean flag rate is 35.42% on the synthetic set against 5.83% in the real study,
and the gap lies in the detector-noise model. It does not support conclusions
about other installations, and it is not a training set.

## Distribution and maintenance

Released as a tagged source archive with a DOI. The pinned instance set and its
content checksum are in `instances/`. Changes to the generator or the noise model
that alter the instance checksum are published as a new version.
