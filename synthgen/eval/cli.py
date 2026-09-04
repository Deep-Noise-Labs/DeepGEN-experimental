"""
``synthgen-eval`` - run the quality gates from the command line.

Two modes:

    # Reference-free: measure a WAV that contains a known sustained note
    synthgen-eval synthesis --audio lead.wav --f0 440.7

    # Reference-based: compare a reconstruction against its source
    synthgen-eval reconstruction --pred out.wav --target in.wav
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .metrics import harmonic_analysis
from .suite import GATES_BY_KEY, SUITE, Result, evaluate_reconstruction


def _read(path: Path) -> tuple[np.ndarray, int]:
    import soundfile as sf

    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return audio.T, sr


def _print_table(results: dict[str, Result]) -> bool:
    width = max(len(r.gate.name) for r in results.values())
    all_passed = True
    print()
    print(f"  {'GATE'.ljust(width)}  {'VALUE':>10}  {'TARGET':>10}  RESULT")
    print(f"  {'-' * width}  {'-' * 10}  {'-' * 10}  ------")
    for result in results.values():
        status = "PASS" if result.passed else "FAIL"
        all_passed &= result.passed
        print(
            f"  {result.gate.name.ljust(width)}  "
            f"{result.value:10.2f}  {result.gate.target:10.2f}  {status}"
            f"   ({result.gate.unit})"
        )
    print()
    return all_passed


def main(argv: list[str] | None = None) -> int:
    # --json lives on a parent parser so it is accepted *after* the
    # subcommand, where anyone would naturally type it. Declaring it on the
    # top-level parser only works before the subcommand, which is a trap.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--json", action="store_true", help="emit JSON instead")

    parser = argparse.ArgumentParser(prog="synthgen-eval", parents=[common])
    sub = parser.add_subparsers(dest="mode", required=True)

    syn = sub.add_parser(
        "synthesis", parents=[common], help="reference-free alias/purity gates"
    )
    syn.add_argument("--audio", type=Path, required=True)
    syn.add_argument("--f0", type=float, required=True, help="fundamental in Hz")

    rec = sub.add_parser(
        "reconstruction", parents=[common], help="reference-based fidelity gates"
    )
    rec.add_argument("--pred", type=Path, required=True)
    rec.add_argument("--target", type=Path, required=True)

    sub.add_parser(
        "gates", parents=[common], help="print the gate definitions and rationale"
    )

    args = parser.parse_args(argv)

    if args.mode == "gates":
        for gate in SUITE:
            print(f"\n{gate.name}  [{gate.key}]")
            print(f"  target  {gate.target} {gate.unit} ({gate.direction})")
            print(f"  stretch {gate.stretch} {gate.unit}")
            print(f"  why     {gate.rationale}")
        return 0

    if args.mode == "synthesis":
        audio, sr = _read(args.audio)
        report = harmonic_analysis(audio, args.f0, sr)
        values = {
            "alias_to_signal_db": report.alias_to_signal_db,
            "sub_fundamental_db": report.sub_fundamental_db,
            "sfdr_db": report.sfdr_db,
        }
        results = {
            k: Result(GATES_BY_KEY[k], v, GATES_BY_KEY[k].passes(v))
            for k, v in values.items()
        }
    else:
        pred, sr = _read(args.pred)
        target, _ = _read(args.target)
        results = evaluate_reconstruction(pred, target, sr)

    if args.json:
        print(
            json.dumps(
                {
                    k: {
                        "value": r.value,
                        "target": r.gate.target,
                        "passed": r.passed,
                        "unit": r.gate.unit,
                    }
                    for k, r in results.items()
                },
                indent=2,
            )
        )
        return 0 if all(r.passed for r in results.values()) else 1

    return 0 if _print_table(results) else 1


if __name__ == "__main__":
    sys.exit(main())
