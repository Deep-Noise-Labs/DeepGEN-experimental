"""
Synthesiser-grade audio evaluation for SynthGen.

``synthgen.eval`` answers a narrower question than the usual text-to-audio
benchmarks: not "does this resemble real audio" but "is this sound clean
enough to put in a session". See :mod:`synthgen.eval.suite` for the gates
and the reasoning behind each threshold.
"""

from .metrics import (
    HarmonicReport,
    alias_to_signal_ratio_db,
    attack_time_ms,
    band_energy_db,
    harmonic_analysis,
    high_frequency_retention_db,
    multires_stft_distance,
    noise_floor_db,
    si_sdr_db,
    spectral_centroid_hz,
    spurious_free_dynamic_range_db,
    stereo_correlation,
    stereo_width_error,
    sub_fundamental_alias_db,
    thd_n_percent,
    transient_error_ms,
)
from .signals import (
    DEFAULT_TEST_FREQS,
    STIMULI,
    SWEEP_TEST_FREQS,
    alias_visibility_hz,
    bandlimited_saw,
    bandlimited_square,
    build_stimulus,
    impulse,
    log_sweep,
    pure_tone,
)
from .suite import (
    SUITE,
    Gate,
    Result,
    evaluate_reconstruction,
    evaluate_synthesis,
)

__all__ = [
    "HarmonicReport",
    "alias_to_signal_ratio_db",
    "attack_time_ms",
    "band_energy_db",
    "harmonic_analysis",
    "high_frequency_retention_db",
    "multires_stft_distance",
    "noise_floor_db",
    "si_sdr_db",
    "spectral_centroid_hz",
    "spurious_free_dynamic_range_db",
    "stereo_correlation",
    "stereo_width_error",
    "sub_fundamental_alias_db",
    "thd_n_percent",
    "transient_error_ms",
    "DEFAULT_TEST_FREQS",
    "SWEEP_TEST_FREQS",
    "alias_visibility_hz",
    "STIMULI",
    "bandlimited_saw",
    "bandlimited_square",
    "build_stimulus",
    "impulse",
    "log_sweep",
    "pure_tone",
    "SUITE",
    "Gate",
    "Result",
    "evaluate_reconstruction",
    "evaluate_synthesis",
]
