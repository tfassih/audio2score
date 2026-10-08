param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Song,
    [string]$Meter = "4/4",
    [switch]$Piano,
    [ValidateSet("auto", "transkun", "spectral")]
    [string]$PianoBackend = "auto",
    [ValidateSet("all", "faithful", "intermediate", "easy")]
    [string]$Arrangement = "all",
    [string]$AnalysisCache = "",
    [switch]$NoEngraving,
    [switch]$NoSeparation
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot
$Launcher = Join-Path $PSScriptRoot ".venv\Scripts\audio2score.exe"
if (-not (Test-Path $Launcher)) {
    throw "audio2score is not installed in .venv. Run .\setup_windows.ps1 first."
}

$argsList = @($Song, "--meter", $Meter)
if ($AnalysisCache) {
    $argsList += @("--analysis-cache", $AnalysisCache, "--arrangement", $Arrangement)
} elseif ($Piano) {
    $argsList += @("--piano", "--piano-backend", $PianoBackend, "--arrangement", $Arrangement)
} elseif (-not $NoSeparation) {
    $argsList += "--separate-vocals"
}
if ($NoEngraving) {
    $argsList += "--no-engraving"
}
& $Launcher @argsList
