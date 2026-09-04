"""
The SynthGen quality gate: a runnable definition of "the job is done".

Scope of the claim
------------------
These thresholds are **proposed engineering targets** for DeepGEN, derived
from perceptual reasoning and standard audio-measurement practice. They are
*not* measured specifications of any third-party product, and nothing here
should be read as a benchmark against a named commercial library. Where a
target is a judgement call it says so.

Why not FAD / CLAP
------------------
Frechet Audio Distance and CLAP score answer "does this resemble the
distribution of real audio / the text prompt". Both are necessary and
neither is sufficient for a *sampler instrument*. A pad can score well on
FAD while aliasing audibly, losing its top octave, and collapsing to mono -
three defects that make it unusable in a session and none of which FAD
penalises. The gates below are the ones a sound designer would fail the
sound on.

Two families of check
---------------------
``synthesis`` gates are measured on the model's own output against a known
stimulus, and need no reference recording. ``reconstruction`` gates compare
output to a reference and are used for codec/VAE work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from . import metrics as M
from .signals import DEFAULT_TEST_FREQS, bandlimited_saw

__all__ = ["Gate", "Result", "SUITE", "evaluate_synthesis", "evaluate_reconstruction"]

Direction = Literal["lower_is_better", "higher_is_better", "closer_to_zero"]


@dataclass(frozen=True)
class Gate:
    """One measurable acceptance criterion."""

    key: str
    name: str
    unit: str
    direction: Direction
    target: float
    """Value at which the gate passes."""
    stretch: float
    """Value that would put DeepGEN clearly ahead of the field."""
    rationale: str

    def passes(self, value: float) -> bool:
        # Cast to native types: metrics return numpy scalars, and a
        # numpy.bool_ leaking out here is not JSON-serialisable downstream.
        value = float(value)
        if self.direction == "lower_is_better":
            return bool(value <= self.target)
        if self.direction == "higher_is_better":
            return bool(value >= self.target)
        return bool(abs(value) <= abs(self.target))


@dataclass
class Result:
    """A gate evaluated against a measurement."""

    gate: Gate
    value: float
    passed: bool
    detail: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.value = float(self.value)
        self.passed = bool(self.passed)


# =============================================================================
# The gates
# =============================================================================

SUITE: tuple[Gate, ...] = (
    Gate(
        key="alias_to_signal_db",
        name="Alias-to-signal ratio",
        unit="dB",
        direction="lower_is_better",
        target=-60.0,
        stretch=-80.0,
        rationale=(
            "Inharmonic energy relative to the note. Below about -60 dB the "
            "folded partials sit under the masking threshold of the note "
            "itself in a mix; above -30 dB they are plainly audible as a "
            "metallic layer. This is the single gate that separates an "
            "oscillator that sounds professional from one that does not."
        ),
    ),
    Gate(
        key="sub_fundamental_db",
        name="Sub-fundamental alias energy",
        unit="dB",
        direction="lower_is_better",
        target=-70.0,
        stretch=-90.0,
        rationale=(
            "Alias products that land *below* the fundamental have nothing "
            "beneath them to mask them, so they are heard directly as grit "
            "or buzz that does not track pitch. Held to a stricter target "
            "than overall aliasing for that reason."
        ),
    ),
    Gate(
        key="sfdr_db",
        name="Spurious-free dynamic range",
        unit="dB",
        direction="higher_is_better",
        target=60.0,
        stretch=80.0,
        rationale=(
            "Fundamental above the single worst inharmonic spur. Catches one "
            "loud whistling artefact that a broadband average would hide."
        ),
    ),
    Gate(
        key="hf_retention_db",
        name="Air-band retention (10-20 kHz)",
        unit="dB",
        direction="closer_to_zero",
        target=1.5,
        stretch=0.5,
        rationale=(
            "The top octave carries the 'expensive' quality of a recorded "
            "library. Losing it is the most common reason neural codec "
            "output sounds dull and small next to the source; gaining it is "
            "not a win either, because added energy up there is noise or "
            "aliasing rather than detail."
        ),
    ),
    Gate(
        key="transient_error_ms",
        name="Attack-time error",
        unit="ms",
        direction="closer_to_zero",
        target=1.0,
        stretch=0.3,
        rationale=(
            "Percussive and plucked instruments are identified largely by "
            "their attack. Around 1 ms of smear is where a pluck starts to "
            "lose its bite; beyond a few ms it stops reading as the source "
            "instrument at all."
        ),
    ),
    Gate(
        key="stereo_width_error",
        name="Stereo image error",
        unit="correlation",
        direction="closer_to_zero",
        target=0.15,
        stretch=0.05,
        rationale=(
            "Codecs trained on mono-heavy data pull the image inwards. A "
            "wide pad that collapses towards the centre loses the exact "
            "property it was chosen for."
        ),
    ),
    Gate(
        key="noise_floor_db",
        name="Noise floor",
        unit="dB",
        direction="lower_is_better",
        target=-70.0,
        stretch=-85.0,
        rationale=(
            "Hiss under a sustained pad accumulates across a layered "
            "arrangement. Commercial sample libraries are effectively "
            "silent between notes; a -50 dB floor is audible on one voice "
            "and unusable across sixteen."
        ),
    ),
    Gate(
        key="si_sdr_db",
        name="Scale-invariant SDR",
        unit="dB",
        direction="higher_is_better",
        target=12.0,
        stretch=20.0,
        rationale=(
            "Waveform-level reconstruction fidelity. A blunt instrument on "
            "its own, but a strong regression tripwire: a codec change that "
            "drops SI-SDR has broken something even if the spectrum looks "
            "fine."
        ),
    ),
    Gate(
        key="multires_stft",
        name="Multi-resolution STFT distance",
        unit="ratio",
        direction="lower_is_better",
        target=0.35,
        stretch=0.15,
        rationale=(
            "Phase-blind timbre distance. Tracks perceived difference far "
            "better than a waveform L1 and is the metric to watch when "
            "tuning a codec's reconstruction loss."
        ),
    ),
)

GATES_BY_KEY = {g.key: g for g in SUITE}


# =============================================================================
# Runners
# =============================================================================


def evaluate_synthesis(
    process: Callable[[np.ndarray], np.ndarray],
    sample_rate: int = 44100,
    freqs: tuple[float, ...] = DEFAULT_TEST_FREQS,
    duration: float = 0.5,
) -> dict[str, Result]:
    """
    Run the reference-free gates on a callable that transforms audio.

    ``process`` takes a mono ``float32`` array and returns one of the same
    length. It can be a whole model, a decoder, or a single module - the
    measurement is valid for anything that claims to pass audio through.

    Each fundamental in ``freqs`` is measured with a band-limited sawtooth
    and the results are aggregated to the *worst* case, because a synth
    that is clean at 110 Hz and dirty at 4 kHz is a synth that is dirty.
    """
    per_freq: dict[float, M.HarmonicReport] = {}
    for f0 in freqs:
        stimulus = bandlimited_saw(f0, duration, sample_rate, 0.5)
        output = np.asarray(process(stimulus), dtype=np.float64)
        per_freq[f0] = M.harmonic_analysis(output, f0, sample_rate)

    worst_asr = max(r.alias_to_signal_db for r in per_freq.values())
    worst_sub = max(r.sub_fundamental_db for r in per_freq.values())
    worst_sfdr = min(r.sfdr_db for r in per_freq.values())
    detail = {
        f"{f0:.1f}Hz": {
            "alias_to_signal_db": r.alias_to_signal_db,
            "sub_fundamental_db": r.sub_fundamental_db,
            "sfdr_db": r.sfdr_db,
        }
        for f0, r in per_freq.items()
    }

    values = {
        "alias_to_signal_db": worst_asr,
        "sub_fundamental_db": worst_sub,
        "sfdr_db": worst_sfdr,
    }
    return {
        key: Result(GATES_BY_KEY[key], value, GATES_BY_KEY[key].passes(value), detail)
        for key, value in values.items()
    }


def evaluate_reconstruction(
    pred: np.ndarray,
    target: np.ndarray,
    sample_rate: int = 44100,
) -> dict[str, Result]:
    """Run the reference-based gates on a reconstruction and its source."""
    values = {
        "hf_retention_db": M.high_frequency_retention_db(pred, target, sample_rate),
        "transient_error_ms": M.transient_error_ms(pred, target, sample_rate),
        "stereo_width_error": M.stereo_width_error(pred, target),
        "noise_floor_db": M.noise_floor_db(pred, sample_rate),
        "si_sdr_db": M.si_sdr_db(pred, target),
        "multires_stft": M.multires_stft_distance(pred, target),
    }
    return {
        key: Result(GATES_BY_KEY[key], value, GATES_BY_KEY[key].passes(value))
        for key, value in values.items()
    }
