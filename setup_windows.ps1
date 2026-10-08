param(
    [switch]$Full,
    [switch]$PianoNeural,
    [switch]$SeparationOnly,
    [string]$PythonVersion = "",
    [switch]$KeepVenv
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Invoke-Checked {
    param(
        [string]$Description,
        [string]$Exe,
        [string[]]$Arguments
    )
    Write-Host ""
    Write-Host $Description
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "$Description failed with exit code $LASTEXITCODE."
    }
}

Write-Host ""
Write-Host "Audio2Score v0.5.1 setup"
Write-Host "========================="
Write-Host ""

# Pick a Python interpreter.  With the Windows Python launcher, -3 selects the
# newest installed Python 3.x.  -PythonVersion lets you explicitly request 3.12
# or 3.13 when you want the most conservative Transkun environment.
$BaseExe = $null
$BaseArgs = @()
if (Get-Command py -ErrorAction SilentlyContinue) {
    $BaseExe = "py"
    if ($PythonVersion) {
        $BaseArgs = @("-$PythonVersion")
    } else {
        $BaseArgs = @("-3")
    }
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    if ($PythonVersion) {
        Write-Warning "The 'py' launcher is not installed, so -PythonVersion cannot select among multiple Pythons. Using 'python' from PATH."
    }
    $BaseExe = (Get-Command python).Source
    $BaseArgs = @()
} else {
    throw "Python was not found. Install 64-bit CPython 3.12 or newer from python.org."
}

$VersionInfo = & $BaseExe @BaseArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}|{64 if sys.maxsize > 2**32 else 32}|{sys.executable}')" 2>$null
if ($LASTEXITCODE -ne 0 -or -not $VersionInfo) {
    throw "Could not start the requested Python interpreter. If you used -PythonVersion, verify it with: py -0p"
}

$Fields = $VersionInfo.Trim() -split '\|'
$DetectedVersion = [version]$Fields[0]
$DetectedBits = $Fields[1]
$DetectedExe = $Fields[2]

if ($DetectedVersion -lt [version]"3.12.0") {
    throw "Audio2Score v0.5.1 requires Python 3.12 or newer. Found $DetectedVersion."
}
if ($DetectedBits -ne "64") {
    throw "Audio2Score requires 64-bit Python. Found $DetectedBits-bit Python at $DetectedExe."
}

Write-Host "Using Python $DetectedVersion"
Write-Host "Interpreter: $DetectedExe"

if ($DetectedVersion.Major -eq 3 -and $DetectedVersion.Minor -eq 14 -and $DetectedVersion.Build -eq 1 -and ($Full -or $SeparationOnly)) {
    throw "audio-separator explicitly excludes Python 3.14.1. Use Python 3.14.2+ or run this setup with -PythonVersion 3.13."
}

if ($DetectedVersion -ge [version]"3.14.0" -and ($PianoNeural -or $Full)) {
    Write-Warning "Transkun can be installed on Python 3.14, but PyTorch warns that torch.jit.script is unsupported on 3.14+. Audio2Score will apply known Transkun compatibility fixes; Python 3.12/3.13 remains the recommended neural-backend environment if TorchScript itself fails."
}

if ((Test-Path ".venv") -and -not $KeepVenv) {
    Write-Host ""
    Write-Host "Removing the existing .venv so stale Python/package state cannot leak into this install..."
    Remove-Item -Recurse -Force ".venv"
}

if (-not (Test-Path ".venv")) {
    $CreateArgs = @() + $BaseArgs + @("-m", "venv", ".venv")
    Invoke-Checked "Creating virtual environment..." $BaseExe $CreateArgs
}

$VenvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    throw "Virtual environment creation failed: $VenvPython does not exist."
}

Invoke-Checked "Updating pip/build tools..." $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel")
Invoke-Checked "Installing Audio2Score core..." $VenvPython @("-m", "pip", "install", "-e", ".")

if ($SeparationOnly) {
    Invoke-Checked "Installing source-separation backend..." $VenvPython @("-m", "pip", "install", "-e", ".[separation]")
}

if ($PianoNeural) {
    Invoke-Checked "Installing Transkun neural piano backend..." $VenvPython @("-m", "pip", "install", "-e", ".[piano-neural]")
}

if ($Full) {
    Invoke-Checked "Installing all optional backends..." $VenvPython @("-m", "pip", "install", "-e", ".[full]")
}

if ($PianoNeural -or $Full) {
    Invoke-Checked "Applying Transkun Python 3.12+ compatibility patch..." $VenvPython @("-m", "audio2score.compat")
}

$Launcher = Join-Path $PSScriptRoot ".venv\Scripts\audio2score.exe"
if (-not (Test-Path $Launcher)) {
    throw "Audio2Score imported but the console launcher was not generated: $Launcher"
}

Invoke-Checked "Running installation doctor..." $VenvPython @("-m", "audio2score.doctor")

Write-Host ""
Write-Host "Installation successful."
Write-Host ""
Write-Host "Activate the environment:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host ""
Write-Host "Verify later at any time:"
Write-Host "  audio2score-doctor"
Write-Host ""
Write-Host "Solo-piano transcription (auto backend):"
Write-Host '  audio2score "C:\path\to\piano.flac" --piano'
Write-Host ""
Write-Host "Force Transkun:"
Write-Host '  audio2score "C:\path\to\piano.flac" --piano --piano-backend transkun'
Write-Host ""
Write-Host "Force the built-in backend:"
Write-Host '  audio2score "C:\path\to\piano.flac" --piano --piano-backend spectral'
