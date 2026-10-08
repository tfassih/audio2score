from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys

from .pipeline import transcribe_song, rerender_from_analysis


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="audio2score",
        description=(
            "Convert recorded music into editable MusicXML/MIDI; "
            "v0.5 adds phrase/dynamic/pedal/section engraving and more playable arrangements."
        ),
    )
    p.add_argument("input", help="Input audio file (FLAC/WAV/MP3/etc.)")
    p.add_argument("-o", "--output-dir", default=None,
                   help="Output directory (default: ./audio2score_output/<song>)")
    p.add_argument("--title", default=None)
    p.add_argument("--composer", default="")
    p.add_argument("--meter", default="4/4", help="Meter, e.g. 4/4 or 3/4")
    p.add_argument("--grid", type=int, choices=[1,2,3,4,6,8], default=4,
                   help="Subdivisions per quarter-note beat; 4 = sixteenths")
    p.add_argument("--melody-backend", choices=["auto","pyin","basic-pitch"], default="auto")
    p.add_argument("--piano", action="store_true",
                   help="Solo-piano mode: polyphonic two-hand grand-staff transcription")
    p.add_argument("--piano-backend", choices=["auto","transkun","spectral"], default="auto",
                   help="Piano engine. auto tries Transkun then the built-in spectral fallback.")
    p.add_argument("--device", choices=["cpu","cuda"], default="cpu",
                   help="Device for optional neural piano backend")
    p.add_argument("--arrangement", choices=["all","faithful","intermediate","easy"],
                   default="all", help="Piano output level; default emits all three arrangements")
    p.add_argument("--no-engraving", action="store_true",
                   help="Disable v0.5 dynamics, phrase slurs, pedal and section markers")
    p.add_argument("--analysis-cache", default=None,
                   help="Reuse a prior Audio2Score analysis JSON instead of retranscribing audio")
    p.add_argument("--separate-vocals", action="store_true",
                   help="Use audio-separator before full-mix analysis")
    p.add_argument("--separator-preset", default="vocal_balanced")
    p.add_argument("--fmin", default="C2")
    p.add_argument("--fmax", default="C7")
    p.add_argument("--no-pdf", action="store_true")
    p.add_argument("--discard-stems", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    input_path = Path(args.input)
    out = Path(args.output_dir) if args.output_dir else Path("audio2score_output") / input_path.stem
    try:
        if args.analysis_cache:
            result = rerender_from_analysis(
                input_path, args.analysis_cache, out,
                arrangement=args.arrangement,
                make_pdf=not args.no_pdf,
                engraving=not args.no_engraving,
                title=args.title,
                composer=args.composer if args.composer else None,
            )
        else:
            result = transcribe_song(
                input_path, out,
                title=args.title, composer=args.composer,
                meter=args.meter, grid=args.grid,
                melody_backend=args.melody_backend,
                separate=args.separate_vocals,
                separator_preset=args.separator_preset,
                fmin=args.fmin, fmax=args.fmax,
                make_pdf=not args.no_pdf,
                keep_stems=not args.discard_stems,
                piano=args.piano,
                piano_backend=args.piano_backend,
                device=args.device,
                arrangement=args.arrangement,
                engraving=not args.no_engraving,
            )
    except Exception as exc:
        logging.exception("Transcription failed") if args.verbose else logging.error(
            "Transcription failed: %s", exc
        )
        return 1
    print(json.dumps(result["outputs"], indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
