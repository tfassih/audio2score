from __future__ import annotations

"""Compatibility helpers for optional third-party transcription backends.

Transkun 2.0.1 predates Python 3.13/3.14.  Its shipped CLI currently relies on
``pkg_resources`` and loads every input with ``pydub.AudioSegment.from_mp3``.
The former disappeared from modern setuptools and the latter pulls in the
removed stdlib ``audioop`` module on Python 3.13+.

Audio2Score patches those two tiny compatibility points in the installed venv:
* package resources are located with pathlib relative to transcribe.py;
* audio is decoded with SoundFile as float32, preserving Transkun's expected
  frames x channels array shape and allowing FLAC/WAV directly.

The patch is deliberately narrow and idempotent.  If a future Transkun version
has already fixed these issues, no source rewrite is performed.
"""

from dataclasses import dataclass
from importlib import metadata, util
from pathlib import Path
import re


@dataclass
class TranskunPatchResult:
    installed: bool
    version: str | None = None
    path: str | None = None
    changed: bool = False
    pkg_resources_fixed: bool = False
    audio_loader_fixed: bool = False
    message: str = ""


def _replace_pkg_resources(text: str) -> tuple[str, bool]:
    if "pkg_resources.resource_filename" not in text:
        return text, False

    # Current Transkun 2.0.1 keeps these statements inside main().  Preserve
    # indentation so the patch remains valid Python.
    pattern = re.compile(
        r"(?m)^(?P<indent>\s*)import pkg_resources\s*\n"
        r"(?P=indent)defaultWeight\s*=\s*\(pkg_resources\.resource_filename\(__name__,\s*[\"']pretrained/2\.0\.pt[\"']\)\)\s*\n"
        r"(?P=indent)defaultConf\s*=\s*\(pkg_resources\.resource_filename\(__name__,\s*[\"']pretrained/2\.0\.conf[\"']\)\)\s*$"
    )

    match = pattern.search(text)
    if not match:
        # Be tolerant of spacing used in the published wheel.
        pattern = re.compile(
            r"(?ms)^(?P<indent>[ \t]*)import pkg_resources\s*\n"
            r"(?P=indent)defaultWeight\s*=.*?pkg_resources\.resource_filename\(__name__,\s*[\"']pretrained/2\.0\.pt[\"']\).*?\n"
            r"(?P=indent)defaultConf\s*=.*?pkg_resources\.resource_filename\(__name__,\s*[\"']pretrained/2\.0\.conf[\"']\).*?(?=\n)"
        )
        match = pattern.search(text)

    if not match:
        raise RuntimeError(
            "Transkun still uses pkg_resources, but Audio2Score could not recognize "
            "the installed transcribe.py layout."
        )

    indent = match.group("indent")
    replacement = (
        f"{indent}from pathlib import Path\n"
        f"{indent}package_dir = Path(__file__).resolve().parent\n"
        f"{indent}defaultWeight = str(package_dir / 'pretrained' / '2.0.pt')\n"
        f"{indent}defaultConf = str(package_dir / 'pretrained' / '2.0.conf')"
    )
    return text[: match.start()] + replacement + text[match.end() :], True


def _replace_audio_loader(text: str) -> tuple[str, bool]:
    # A future Transkun release may already use soundfile/torchaudio.
    if "import pydub" not in text and "AudioSegment.from_mp3" not in text:
        return text, False

    pattern = re.compile(
        r"(?ms)^def readAudio\(path,\s*normalize\s*=\s*True\):\s*\n"
        r".*?^\s*return\s+audio\.frame_rate,\s*y\s*$"
    )
    match = pattern.search(text)
    if not match:
        raise RuntimeError(
            "Transkun still uses pydub for audio loading, but Audio2Score could "
            "not recognize the installed readAudio() implementation."
        )

    replacement = '''def readAudio(path, normalize=True):
    # Audio2Score compatibility patch: Python 3.13 removed audioop, while the
    # published Transkun 2.0.1 loader imports pydub and forces from_mp3() even
    # for FLAC.  SoundFile already ships with Audio2Score and returns normalized
    # float audio in Transkun's expected frames x channels shape.
    import soundfile as sf
    y, fs = sf.read(path, dtype="float32", always_2d=True)
    return fs, y'''
    return text[: match.start()] + replacement + text[match.end() :], True


def patch_transkun() -> TranskunPatchResult:
    spec = util.find_spec("transkun")
    if spec is None or spec.origin is None:
        return TranskunPatchResult(
            installed=False,
            message="Transkun is not installed.",
        )

    try:
        version = metadata.version("transkun")
    except metadata.PackageNotFoundError:
        version = "unknown"

    package_dir = Path(spec.origin).resolve().parent
    target = package_dir / "transcribe.py"
    if not target.exists():
        raise RuntimeError(f"Could not locate Transkun transcribe.py at {target}")

    original = target.read_text(encoding="utf-8")
    updated, fixed_pkg = _replace_pkg_resources(original)
    updated, fixed_audio = _replace_audio_loader(updated)
    changed = updated != original

    if changed:
        backup = target.with_suffix(".py.audio2score-original")
        if not backup.exists():
            backup.write_text(original, encoding="utf-8")
        target.write_text(updated, encoding="utf-8")

    # Verify that both old failure paths are gone.  A future fixed version may
    # naturally satisfy this without requiring a source edit.
    final = target.read_text(encoding="utf-8")
    if "pkg_resources.resource_filename" in final:
        raise RuntimeError("Transkun pkg_resources compatibility patch did not apply.")
    if "AudioSegment.from_mp3" in final or "import pydub" in final:
        raise RuntimeError("Transkun audio-loader compatibility patch did not apply.")

    return TranskunPatchResult(
        installed=True,
        version=version,
        path=str(target),
        changed=changed,
        pkg_resources_fixed=fixed_pkg,
        audio_loader_fixed=fixed_audio,
        message=(
            "Transkun compatibility patch applied."
            if changed
            else "Transkun is already compatible/patched."
        ),
    )


def main() -> int:
    result = patch_transkun()
    print(result.message)
    if result.installed:
        print(f"Transkun version: {result.version}")
        print(f"Transkun source:  {result.path}")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
