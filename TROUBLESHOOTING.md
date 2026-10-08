# Audio2Score v0.6.0 troubleshooting

## `audio2score` is not recognized

Run setup from this repository:

```powershell
.\setup_windows.ps1
.\.venv\Scripts\Activate.ps1
where.exe audio2score
audio2score-doctor
```

The setup script now fails explicitly if `.venv\\Scripts\\audio2score.exe` was
not generated.

## Transkun says `No module named pkg_resources`

v0.6.0 patches this automatically. Run:

```powershell
python -m audio2score.compat
audio2score-doctor
```

If you replaced/reinstalled Transkun after setup, the first Audio2Score
Transkun run will patch it again automatically.

## Transkun/pydub says `No module named audioop` or `pyaudioop`

Audio2Score's Transkun path no longer uses pydub after the automatic patch.
Run:

```powershell
python -m audio2score.compat
audio2score-doctor
```

The neural install also includes `audioop-lts` on Python 3.13+ as a fallback.

## Python 3.14 prints a `torch.jit.script` FutureWarning

That warning is upstream in PyTorch/Transkun.  It is not the old pkg_resources
or audioop failure.  If transcription completes, it is only a warning.  If it
fails inside TorchScript, keep Python 3.14 installed but build this project's
venv with Python 3.13:

```powershell
py -0p
.\setup_windows.ps1 -PianoNeural -PythonVersion 3.13
```

## Check the complete environment

```powershell
audio2score-doctor
```
