"""Objective audio-quality evaluation for SynthGen.

See ``docs/EVALS.md`` for what each metric means and the thresholds a
checkpoint must clear before it is considered sample-library grade.
"""

from synthgen.eval.metrics import (
    alias_to_signal_ratio,
    band_energy_error_db,
    crest_factor_db,
    harmonic_partial_levels,
    log_spectral_distance,
    stereo_correlation,
    total_harmonic_distortion,
)
from synthgen.eval.probes import (
    harmonic_sweep,
    pluck,
    saw_note,
    sine,
)

__all__ = [
    "alias_to_signal_ratio",
    "band_energy_error_db",
    "crest_factor_db",
    "harmonic_partial_levels",
    "log_spectral_distance",
    "stereo_correlation",
    "total_harmonic_distortion",
    "harmonic_sweep",
    "pluck",
    "saw_note",
    "sine",
]
