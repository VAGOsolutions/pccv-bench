"""PCCV-Bench — a public, fully synthetic benchmark for process-conditioned
consistency verification.

Contains zero real factory geometry. Every scene, camera and detection is
generated; the only thing this package shares with the real study is the
*code path* — it drives the exact same production matcher
(``pccv_bench.core.matching.auto_match_boxes_to_parts``) and the exact
same metric/ablation layer (``pccv_bench.core.harness`` / ``.evaluate``) that
``eval/injection/real_provider.py`` drives on real data. That shared code path
is what lets a reproducibility claim ("same mechanism, two data sources") be
checked rather than asserted.
"""
