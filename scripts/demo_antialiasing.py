#!/usr/bin/env python3
"""
Render before/after audio for the decoder anti-aliasing change.

What this does and does NOT show
--------------------------------
This is a *mechanism* demo, not a comparison of two trained checkpoints. It runs
real audio through the decoder's nonlinearity chain with the anti-aliasing
sandwich off ("before") and on ("after"), using the exact modules from
``synthgen.model.activations``. Nothing else differs between the two renders, so
every difference you hear is aliasing and only aliasing.

It is not a checkpoint comparison because training a 44.1 kHz VAE to convergence
is a multi-GPU-day job. What it establishes is the thing a checkpoint comparison
could not: that the artefact is structural, present at every nonlinearity site,
and not something more training would have fixed.

Standing in for the convolutions
--------------------------------
The decoder's final stage runs roughly eight Snake sites at full 44.1 kHz rate
(one before the transposed convolution, two per residual block, one at the output
projection). The learned convolutions between them are not available without a
trained checkpoint, so this chain applies the activations back to back with a
DC-removal and RMS-match between sites, standing in for the weight-normalised
convolutions and residual connections that would otherwise hold the signal level
steady. Both renders get identical treatment.

Usage:
    python scripts/demo_antialiasing.py --output-dir demo_out
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from synthgen.model.activations import AntiAliasedActivation, Snake
from synthgen.training.losses import MultiResolutionSTFTLoss

SAMPLE_RATE = 44100

# Eight activation sites, matching the density of the decoder's full-rate stage.
FULL_RATE_SITES = 8
# Modest alpha: a trained decoder's weight-normalised convolutions keep
# activations in a moderate range, so the per-site nonlinearity is mild rather
# than the hard distortion a large alpha would produce.
ALPHA = 0.9


# =============================================================================
# The chain
# =============================================================================


def _normalise(y: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Remove the DC that ``sin^2`` introduces, then match the input's RMS."""
    y = y - y.mean(dim=-1, keepdim=True)
    scale = reference.std() / (y.std() + 1e-9)
    return y * scale


def decoder_tail(
    audio: torch.Tensor,
    anti_aliased: bool,
    sites: int = FULL_RATE_SITES,
    alpha: float = ALPHA,
) -> torch.Tensor:
    """Run ``sites`` Snake activations over the signal, with or without the sandwich."""
    activation = Snake(1, alpha_init=alpha)
    activation.alpha.data.fill_(alpha)
    module = AntiAliasedActivation(activation, ratio=2) if anti_aliased else activation

    x = audio
    with torch.no_grad():
        for _ in range(sites):
            x = _normalise(module(x), x)
    return x


# =============================================================================
# Measurement
# =============================================================================


def _band_energy(signal: torch.Tensor, ceiling_hz: float) -> float:
    """Total energy below ``ceiling_hz``."""
    y = signal.flatten()
    spectrum = torch.fft.rfft(y * torch.hann_window(y.numel())).abs() ** 2
    freqs = torch.fft.rfftfreq(y.numel(), 1.0 / SAMPLE_RATE)
    return spectrum[freqs < ceiling_hz].sum().item()


def alias_to_signal_db(
    before: torch.Tensor,
    after: torch.Tensor,
    ceiling_hz: float = 18000.0,
) -> float:
    """
    Level of what the anti-aliasing removed, relative to the clean render.

    Measured below 18 kHz on purpose. The decimation filter is flat to 20 kHz but
    necessarily rolls off through the last 2 kHz to Nyquist, so energy removed up
    there is partly legitimate content, not fold-back. Below 18 kHz the filter is
    transparent to within 0.02 dB, so everything the two renders differ by is
    aliasing.
    """
    difference = _band_energy(before - after, ceiling_hz)
    signal = _band_energy(after, ceiling_hz)
    return 10 * math.log10((difference + 1e-20) / (signal + 1e-20))


def harmonic_purity_db(
    signal: torch.Tensor,
    fundamental: float,
    sample_rate: int = SAMPLE_RATE,
    tolerance_hz: float = 12.0,
) -> float:
    """
    Ratio of harmonic to inharmonic energy, in dB. Higher is cleaner.

    Only meaningful for a steady tone at a known fundamental: every spectral peak
    that is not within ``tolerance_hz`` of an integer multiple of ``fundamental``
    is energy that does not belong in the sound.
    """
    y = signal.flatten()
    spectrum = torch.fft.rfft(y * torch.hann_window(y.numel())).abs() ** 2
    freqs = torch.fft.rfftfreq(y.numel(), 1.0 / sample_rate)

    nearest_harmonic = torch.round(freqs / fundamental) * fundamental
    is_harmonic = (freqs - nearest_harmonic).abs() <= tolerance_hz
    is_harmonic &= freqs > fundamental / 2  # exclude the DC region

    harmonic = spectrum[is_harmonic].sum().item()
    inharmonic = spectrum[~is_harmonic].sum().item()
    return 10 * math.log10((harmonic + 1e-20) / (inharmonic + 1e-20))


def log_spectrum(
    audio: torch.Tensor,
    points: int = 160,
    fmin: float = 30.0,
    fmax: float = 22050.0,
) -> list[float]:
    """
    Average magnitude spectrum in dB, resampled onto a log-frequency grid.

    Log frequency because that is how pitch works and how a spectrum should be
    read; a linear axis buries everything below 2 kHz in the first tenth of the
    plot.
    """
    y = audio.flatten()
    window = torch.hann_window(4096)
    spec = torch.stft(y, 4096, 2048, 4096, window, return_complex=True).abs()
    mean = spec.mean(dim=-1)
    freqs = torch.fft.rfftfreq(4096, 1.0 / SAMPLE_RATE)

    edges = torch.logspace(math.log10(fmin), math.log10(fmax), points + 1)
    out: list[float] = []
    for low, high in zip(edges[:-1], edges[1:]):
        mask = (freqs >= low) & (freqs < high)
        if mask.any():
            value = mean[mask].max()
        else:
            # At the bottom of a log axis a band can be narrower than the FFT's
            # bin spacing. Fall back to the nearest bin rather than a hole.
            centre = (low * high).sqrt()
            value = mean[int(torch.argmin((freqs - centre).abs()))]
        out.append(round(20 * math.log10(float(value) + 1e-10), 2))
    return out


def log_spectrum_axis(
    points: int = 160, fmin: float = 30.0, fmax: float = 22050.0
) -> list[float]:
    """Band centre frequencies matching :func:`log_spectrum`."""
    edges = torch.logspace(math.log10(fmin), math.log10(fmax), points + 1)
    return [round(float((a * b).sqrt()), 1) for a, b in zip(edges[:-1], edges[1:])]


def phase_scrambled(audio: torch.Tensor, seed: int = 0) -> torch.Tensor:
    """
    A signal with (almost) the same magnitude spectrogram and destroyed phase.

    Used to show what a magnitude-only objective cannot see: this scores well on
    a multi-resolution magnitude STFT loss and sounds like a wash of noise.
    """
    generator = torch.Generator().manual_seed(seed)
    y = audio.flatten()
    window = torch.hann_window(2048)
    spec = torch.stft(y, 2048, 512, 2048, window, return_complex=True)
    random_phase = torch.rand(spec.shape, generator=generator) * 2 * math.pi
    scrambled = spec.abs() * torch.exp(1j * random_phase)
    out = torch.istft(scrambled, 2048, 512, 2048, window, length=y.numel())
    return out.view_as(audio)


# =============================================================================
# Sources
# =============================================================================


def load_clip(
    path: Path,
    seconds: float,
    offset_seconds: float = 0.0,
) -> torch.Tensor:
    """Load a mono clip at its native rate. Refuses anything that is not 44.1 kHz."""
    info = sf.info(str(path))
    if info.samplerate != SAMPLE_RATE:
        raise ValueError(
            f"{path.name} is {info.samplerate} Hz; resampling would band-limit the "
            "source and change the very artefact under test. Use a 44.1 kHz source."
        )

    start = min(int(offset_seconds * SAMPLE_RATE), max(info.frames - 1, 0))
    audio, _ = sf.read(
        str(path),
        start=start,
        stop=min(start + int(seconds * SAMPLE_RATE), info.frames),
        dtype="float32",
        always_2d=True,
    )
    if audio.size == 0:
        raise ValueError(f"{path.name}: no audio at offset {offset_seconds}s")

    mono = audio.mean(axis=1)
    mono = mono / (np.abs(mono).max() + 1e-9) * 0.7
    return torch.from_numpy(mono).view(1, 1, -1)


def synthetic_saw(
    fundamental: float = 220.0,
    seconds: float = 2.5,
    cutoff_hz: float = 9000.0,
) -> torch.Tensor:
    """
    A band-limited sawtooth: a clean, exactly-known synthesiser waveform.

    Band-limited on purpose. Every partial is below Nyquist, so the source itself
    contributes no aliasing and any inharmonic content in the render came from
    the chain.
    """
    t = torch.arange(int(seconds * SAMPLE_RATE), dtype=torch.float32) / SAMPLE_RATE
    wave = torch.zeros_like(t)
    harmonic = 1
    while harmonic * fundamental < cutoff_hz:
        wave += torch.sin(2 * math.pi * harmonic * fundamental * t) / harmonic
        harmonic += 1
    wave = wave / wave.abs().max() * 0.7
    return wave.view(1, 1, -1)


# =============================================================================
# Entry point
# =============================================================================


def write(path: Path, audio: torch.Tensor, gain: float = 1.0) -> None:
    sf.write(str(path), audio.flatten().numpy() * gain, SAMPLE_RATE, subtype="PCM_16")


def match_gain(*signals: torch.Tensor, peak: float = 0.89) -> float:
    """
    One gain for a whole comparison group.

    A/B listening is only meaningful at matched level -- normalising each render
    to its own peak would make the louder one sound better regardless of what is
    actually in it.
    """
    highest = max(signal.abs().max().item() for signal in signals)
    return peak / (highest + 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("demo_out"))
    parser.add_argument(
        "--source",
        action="append",
        default=None,
        metavar="PATH[:OFFSET[:SECONDS]]",
        help="44.1 kHz audio file to run through the chain, with an optional "
        "start offset and length in seconds. Repeatable.",
    )
    parser.add_argument("--seconds", type=float, default=2.5)
    parser.add_argument("--offset", type=float, default=0.0)
    args = parser.parse_args()

    def parse_source(spec: str) -> tuple[Path, float, float]:
        parts = spec.split(":")
        path = Path(parts[0])
        offset = float(parts[1]) if len(parts) > 1 and parts[1] else args.offset
        seconds = float(parts[2]) if len(parts) > 2 and parts[2] else args.seconds
        return path, offset, seconds

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict] = {}

    sources: list[tuple[str, torch.Tensor, float | None]] = [
        ("saw_220hz", synthetic_saw(220.0, args.seconds), 220.0),
    ]
    for spec in args.source or []:
        path, offset, seconds = parse_source(spec)
        sources.append((path.stem, load_clip(path, seconds, offset), None))

    stft_loss = MultiResolutionSTFTLoss()
    spectra: dict[str, dict] = {"axis_hz": log_spectrum_axis()}

    for name, reference, fundamental in sources:
        before = decoder_tail(reference, anti_aliased=False)
        after = decoder_tail(reference, anti_aliased=True)

        removed = before - after
        # before/after share a gain so the A/B is level-matched; the residual is
        # far too quiet to hear at that gain, so it gets its own, reported below.
        pair_gain = match_gain(before, after)
        removed_gain = match_gain(removed)

        write(args.output_dir / f"{name}__00_reference.wav", reference, match_gain(reference))
        write(args.output_dir / f"{name}__01_before.wav", before, pair_gain)
        write(args.output_dir / f"{name}__02_after.wav", after, pair_gain)
        write(args.output_dir / f"{name}__03_removed.wav", removed, removed_gain)

        entry: dict[str, float] = {
            "alias_to_signal_db": alias_to_signal_db(before, after),
            "residual_listening_gain_db": 20 * math.log10(removed_gain / pair_gain),
            "stft_loss_before": float(stft_loss(before, reference)),
            "stft_loss_after": float(stft_loss(after, reference)),
        }
        if fundamental is not None:
            entry["harmonic_purity_before_db"] = harmonic_purity_db(before, fundamental)
            entry["harmonic_purity_after_db"] = harmonic_purity_db(after, fundamental)
        report[name] = entry

        spectra[name] = {
            "reference": log_spectrum(reference),
            "before": log_spectrum(before),
            "after": log_spectrum(after),
            "removed": log_spectrum(before - after),
        }

    # What a magnitude-only objective cannot see.
    probe_name, probe, _ = sources[-1]
    scrambled = phase_scrambled(probe)
    write(
        args.output_dir / f"{probe_name}__04_phase_scrambled.wav",
        scrambled,
        match_gain(scrambled),
    )
    report["magnitude_objective_blind_spot"] = {
        "source": probe_name,
        "stft_loss_phase_scrambled": float(stft_loss(scrambled, probe)),
        "stft_loss_naive_chain": float(stft_loss(decoder_tail(probe, False), probe)),
    }

    with open(args.output_dir / "report.json", "w") as handle:
        json.dump(report, handle, indent=2)
    with open(args.output_dir / "spectra.json", "w") as handle:
        json.dump(spectra, handle)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
