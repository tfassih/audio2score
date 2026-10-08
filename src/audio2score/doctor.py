from __future__ import annotations

from importlib import metadata, util
from pathlib import Path
import shutil
import subprocess
import sys

from .compat import patch_transkun


def _version(name: str) -> str:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "not installed"


def main() -> int:
    print("Audio2Score doctor")
    print("==================")
    print(f"Python:       {sys.version.split()[0]}")
    print(f"Executable:   {sys.executable}")
    print(f"Audio2Score:  {_version('audio2score')}")
    print(f"NumPy:        {_version('numpy')}")
    print(f"SciPy:        {_version('scipy')}")
    print(f"librosa:      {_version('librosa')}")
    print(f"SoundFile:    {_version('soundfile')}")
    print(f"pretty_midi:  {_version('pretty_midi')}")
    print(f"PyTorch:      {_version('torch')}")
    print(f"Transkun:     {_version('transkun')}")
    print(f"FFmpeg:       {shutil.which('ffmpeg') or 'not found (optional for many formats)'}")

    if sys.version_info < (3, 12):
        print("ERROR: Python 3.12 or newer is required.")
        return 1

    if util.find_spec("transkun") is not None:
        try:
            result = patch_transkun()
            print(f"Transkun fix: {result.message}")
            cmd = [sys.executable, "-m", "transkun.transcribe", "--help"]
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if proc.returncode != 0:
                print("ERROR: Transkun help/self-test failed:")
                print((proc.stderr or proc.stdout).strip())
                return 2
            print("Transkun CLI: OK")
            if sys.version_info >= (3, 14):
                print(
                    "NOTE: PyTorch currently warns that torch.jit.script is not "
                    "supported on Python 3.14+. Transkun may still run, but its "
                    "neural backend is considered experimental on Python 3.14."
                )
        except Exception as exc:
            print(f"ERROR: Transkun compatibility check failed: {exc}")
            return 2

    launcher = Path(sys.executable).parent / ("audio2score.exe" if sys.platform == "win32" else "audio2score")
    print(f"Launcher:     {launcher if launcher.exists() else 'not found'}")
    if not launcher.exists():
        print("ERROR: console launcher is missing. Re-run setup_windows.ps1.")
        return 3

    print("\nCore installation looks good.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
