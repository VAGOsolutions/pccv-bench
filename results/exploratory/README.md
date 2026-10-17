# Exploratory noise ablation (not pre-registered)

Two seed-42 medium-tier runs with parts of the v1.2-fitted detector-noise model
switched off, scored at the frozen operating point (`ablate --frozen-operating-point`).
They are **exploratory**: planned after the E0-E7 results were seen, to locate the
source of the clean-rate gap. A full sensitivity study is future work.

| Run | Change from `medium_v1_2_fitted.yaml` | Config hash |
|---|---|---|
| `nomiss` | pooled `miss_rate` 0 | `6c31c3587a597653` |
| `nodetnoise` | miss, FP, merge, split rates 0; jitter 0 (pooled and per camera) | `6570b374ab2e9c2e` |

The configs are `medium_v1_2_<run>.yaml`; the same config is embedded in each result
file's `generation_meta.config`. Instance manifests are included; instance files are
not (regenerate with
`python -m pccv_bench.cli generate --config <yaml> --seed 42 --out <path>`).

Headline (frozen verifier `+all+twoway+offset_aware`, clean flag rate): fitted 35.42%,
no misses 23.75%, no detector noise 1.25%. These were regenerated from this repository
and match the original runs exactly (the hashes differ from the originals only because
the camera identifiers changed).
