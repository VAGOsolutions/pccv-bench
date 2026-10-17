"""Observation-model noise parameters.

**Versioning discipline:** this ships as ``v0.1`` with
hand-set, plausible parameters so the generator is testable end-to-end
immediately. Refitting to measured detector statistics (per-camera
miss/FP/jitter/split/merge rates) is future work, to be published as ``v1.0``
once done — the paper must report v1.0 only, and every result file must carry
this version string so the provenance is traceable.
"""
from __future__ import annotations

from dataclasses import dataclass

NOISE_VERSION = "v0.1-hand-set"


@dataclass(frozen=True)
class NoiseParams:
    """Per-camera detection-noise parameters .

    All rates are probabilities per *group* (or, for ``fp_rate``, per image),
    applied once per instance so a single group is never hit by more than one
    of miss/split/merge in a way that would double-perturb it.
    """

    version: str = NOISE_VERSION
    miss_rate: float = 0.03          # a group produces no detection at all
    fp_rate: float = 0.03            # a spurious extra box appears somewhere
    fp_size_frac: tuple[float, float] = (0.01, 0.05)  # FP box area, frac of frame
    jitter_px_sigma: float = 4.0     # per-edge Gaussian localisation noise
    split_rate: float = 0.05         # one true group -> two adjacent boxes
    merge_rate: float = 0.05         # two true groups -> one enclosing box
    score_mean: float = 0.75
    score_sigma: float = 0.12

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "miss_rate": self.miss_rate,
            "fp_rate": self.fp_rate,
            "fp_size_frac": list(self.fp_size_frac),
            "jitter_px_sigma": self.jitter_px_sigma,
            "split_rate": self.split_rate,
            "merge_rate": self.merge_rate,
            "score_mean": self.score_mean,
            "score_sigma": self.score_sigma,
        }


#: Default v0.1 parameters, identical across cameras (no measured per-camera
#: detector statistics exist yet to differentiate them — see the module
#: docstring). A config may override per camera.
DEFAULT_NOISE = NoiseParams()

#: Per-camera noise convention (per-camera pass): a caller that has camera-specific measurements supplies a
#: ``dict[str, NoiseParams]`` keyed by camera name wherever this module's
#: consumers (``SyntheticProvider``) accept ``NoiseParams | dict[str,
#: NoiseParams]``. Any camera key *absent* from that dict falls back to
#: :data:`DEFAULT_NOISE` — the same "pooled/default" params this module
#: already ships as the single-``NoiseParams`` default — not an error, since
#: a caller building the dict from a subset of cameras (e.g. an ablation
#: over fewer stations) should not have to enumerate every camera that
#: exists. Passing a plain ``NoiseParams`` (as every caller in this
#: codebase did before this pass) still means exactly what it always meant:
#: that one set of parameters, used for every camera.
