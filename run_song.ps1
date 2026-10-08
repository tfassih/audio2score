param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Song,
    [string]$Meter = "4/4",
    [switch]$Piano,
    [ValidateSet("auto", "transkun", "spectral")]
    [string]$PianoBackend = "auto",
    [ValidateSet("all", "faithful", "intermediate", "easy")]
    [string]$Arrangement = "all",
    [ValidateSet("adaptive", "fixed")]
    [string]$Quantizer = "adaptive",
    [ValidateSet("conservative", "balanced", "aggressive")]
    [string]$ValidationStrength = "balanced",
    [string]$AnalysisCache = "",
    [string]$PianoMidiInput = "",
    [switch]$NoValidation,
    [switch]$NoAddMissing,
    [switch]$NoPitchCorrection,
    [switch]$NoEngraving,
    [switch]$NoSeparation
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot ".venv\Scripts\audio2score.exe"
if (-not (Test-Path $Launcher)) {
    throw "audio2score is not installed in .venv. Run .\setup_windows.ps1 first."
}

$argsList = @($Song, "--meter", $Meter, "--quantizer", $Quantizer)
if ($AnalysisCache) {
    $argsList += @("--analysis-cache", $AnalysisCache, "--arrangement", $Arrangement)
} elseif ($Piano) {
    $argsList += @(
        "--piano",
        "--piano-backend", $PianoBackend,
        "--arrangement", $Arrangement,
        "--validation-strength", $ValidationStrength
    )
    if ($PianoMidiInput) { $argsList += @("--piano-midi-input", $PianoMidiInput) }
    if ($NoValidation) { $argsList += "--no-validation" }
    if ($NoAddMissing) { $argsList += "--no-add-missing" }
    if ($NoPitchCorrection) { $argsList += "--no-pitch-correction" }
} elseif (-not $NoSeparation) {
    $argsList += "--separate-vocals"
}
if ($NoEngraving) {
    $argsList += "--no-engraving"
}
& $Launcher @argsList
