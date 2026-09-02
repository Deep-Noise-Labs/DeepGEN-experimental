"""
Deterministic probe signals for measuring a signal path.

These are measurement instruments, not model output. They are synthesised
analytically so that the *correct* answer is known exactly: a band-limited saw
at f0 has energy at integer multiples of f0 and nowhere else, so any other
energy downstream is unambiguously an artefact introduced by the system under
test. That property is what makes the aliasing measurement falsifiable.

Every generator is seeded/closed-form and therefore bit-reproducible.
"""

from __future__ import annotations

import numpy as np


def sine(f0: float, duration: float, sample_rate: int = 44100) -> np.ndarray:
    """A pure sine. The cleanest possible aliasing probe."""
    t = np.arange(int(duration * sample_rate)) / sample_rate
    return np.sin(2.0 * np.pi * f0 * t).astype(np.float32)


def saw_note(
    f0: float,
    duration: float,
    sample_rate: int = 44100,
    max_partials: int | None = None,
) -> np.ndarray:
    """
    An *additively band-limited* sawtooth -- the canonical synth waveform.

    Built by summing partials only up to Nyquist rather than by naive
    wrapping, so the probe itself contributes zero aliasing and any alias
    energy measured downstream belongs to the system under test.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    nyquist = sample_rate / 2.0
    limit = int(nyquist // f0) if max_partials is None else max_partials

    out = np.zeros_like(t)
    for order in range(1, limit + 1):
        if order * f0 >= nyquist:
            break
        out += ((-1.0) ** (order + 1)) * np.sin(2.0 * np.pi * order * f0 * t) / order
    out *= 2.0 / np.pi
    peak = np.max(np.abs(out))
    return (out / peak if peak > 0 else out).astype(np.float32)


def pluck(
    f0: float,
    duration: float,
    sample_rate: int = 44100,
    decay: float = 6.0,
) -> np.ndarray:
    """
    A decaying band-limited harmonic tone with a fast attack.

    Stresses transient behaviour and the high-order partials that a pluck
    excites at onset -- the hardest case for an aliasing signal path.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    nyquist = sample_rate / 2.0

    out = np.zeros_like(t)
    order = 1
    while order * f0 < nyquist:
        # Higher partials decay faster, as on a real plucked string.
        envelope = np.exp(-decay * t * (1.0 + 0.35 * (order - 1)))
        out += envelope * np.sin(2.0 * np.pi * order * f0 * t) / order
        order += 1

    attack = np.minimum(1.0, t / 0.003)  # 3 ms attack
    out *= attack
    peak = np.max(np.abs(out))
    return (out / peak if peak > 0 else out).astype(np.float32)


def harmonic_sweep(
    f_start: float,
    f_end: float,
    duration: float,
    sample_rate: int = 44100,
) -> np.ndarray:
    """
    An exponential sine sweep.

    On a spectrogram a linear system shows a single clean diagonal. An
    aliasing system shows extra diagonals travelling *downwards* as the sweep
    rises -- harmonics reflecting off Nyquist. This is the most legible
    visual signature of aliasing, which is why it is included as a probe.
    """
    t = np.arange(int(duration * sample_rate)) / sample_rate
    ratio = f_end / f_start
    phase = 2.0 * np.pi * f_start * duration / np.log(ratio) * (ratio ** (t / duration) - 1.0)
    return np.sin(phase).astype(np.float32)
