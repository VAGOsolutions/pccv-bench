# Detector-noise model

`pccv_bench/observe.py` turns each scene's truth boxes into noisy detections
with five mechanisms plus occlusion. The `v1.2-fitted` parameters in
`pccv_bench/configs/medium_v1_2_fitted.yaml` were measured from one frozen
detector's raw predictions on a chronological test split of the real
installation. That raw data is not released; the fitted values are, together
with the definitions below so a reviewer can judge them and so another
installation's statistics can be swapped in.

## Mechanisms

| Parameter | Mechanism |
|---|---|
| `miss_rate` | A truth group with no usable detection. |
| `fp_rate` | At most one spurious box per image. |
| `fp_size_frac` | (low, high) fraction of the image area a spurious box covers, sampled uniformly. |
| `jitter_px_sigma` | One scalar standard deviation, applied independently to all four box edges. |
| `split_rate` | Probability a truth group is reported as two or more boxes. |
| `merge_rate` | Probability a truth group is merged with the next one. |
| `score_mean`, `score_sigma` | Detection score. Hand-set, because nothing downstream consumes it. |

## Fitting definitions

These were chosen because the observation model needs exactly these numbers,
not because they are the only reasonable choices.

- Raw detections are first clustered by merging only literally touching or
  overlapping boxes (gap 0), the same predicted-box units the frozen
  recall and precision numbers are computed from.
- **Jitter.** For every clustered box matched 1:1 to a truth box at
  IoU >= 0.5 (greedy, highest IoU first), the four signed per-edge pixel
  deltas (predicted edge minus truth edge) are pooled into one sample, and
  `jitter_px_sigma` is their standard deviation.
- **Split.** A truth box left unmatched by the 1:1 pass, whose area is at least
  50% covered by each of two or more still-unmatched clustered boxes whose
  union reaches IoU >= 0.5 with it, counts as one split event.
- **Merge.** Symmetric to split: a clustered box left unmatched by the 1:1
  pass and by any split, which covers at least 50% of two or more truth
  boxes, counts as one merge event.
- **Rates.** `split_rate` and `merge_rate` are each (event count) / (total
  truth boxes). This slightly overcounts merge relative to the generator's
  single per-group draw, because each merge event consumes two groups' worth
  of slots.
- **False-positive size.** Every clustered box left after miss, split and
  merge accounting is a false positive. `fp_size_frac` is the (10th, 90th)
  percentile of its area as a fraction of the image area.
- **miss_rate and fp_rate** come from the frozen evaluation's summary:
  one minus recall, and the empty-image false-positive rate
  (`pccv_bench/fit_noise.py`).

## Per-camera fits and fallback

Each of the four fitted mechanisms (jitter, split, merge, false-positive
size) is also fitted per camera and checked against a minimum-sample
threshold. A camera-mechanism pair below its threshold is measured but not
trusted over the pooled fit: it falls back to the pooled value. The
per-camera comments in `medium_v1_2_fitted.yaml` state, camera by camera,
which pairs used their own estimate and which fell back and why. A camera
absent from `per_camera_noise` falls back to the pooled block.

## Tiers

`easy_v1_2.yaml` and `hard_v1_2.yaml` are derived from the medium tier by
`pccv_bench/make_tier_configs.py`: jitter, miss, false-positive and merge
rates scaled by 0.5 and 1.5 (capped at 1), occlusion 0.05 and 0.20.
