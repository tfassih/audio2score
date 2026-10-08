param(
    [switch]$Full,
    [switch]$PianoNeural,
    [switch]$SeparationOnly
)

$ErrorActionPreference = "Stop"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install 64-bit Python 3.11 from python.org."
}
$ver = & py -3.11 -c "import sys; print('.'.join(map(str, sys.version_info[:2])))" 2>$null
if ($LASTEXITCODE -ne 0 -or $ver -ne "3.11") {
    throw "Python 3.11 is required for the optional ML stack."
}
& py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel
if ($SeparationOnly) {
    & .\.venv\Scripts\python.exe -m pip install -e ".[separation]"
} elseif ($Full) {
    & .\.venv\Scripts\python.exe -m pip install -e ".[full]"
} else {
    & .\.venv\Scripts\python.exe -m pip install -e .
}
if ($PianoNeural) {
    Write-Host "Installing optional Transkun piano backend..."
    & .\.venv\Scripts\python.exe -m pip install transkun
}
Write-Host ""
Write-Host "Installed. Activate with:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Solo-piano transcription:"
Write-Host '  audio2score "C:\path\to\piano.flac" --piano'
Write-Host ""
Write-Host "Built-in spectral piano mode always works with the baseline install."
Write-Host "-PianoNeural adds Transkun when its current package supports your environment."
