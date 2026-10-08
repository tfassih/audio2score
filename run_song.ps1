param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$Song,
    [string]$Meter = "4/4",
    [switch]$Piano,
    [ValidateSet("all", "faithful", "intermediate", "easy")]
    [string]$Arrangement = "all",
    [string]$AnalysisCache = "",
    [switch]$NoEngraving,
    [switch]$NoSeparation
)
$ErrorActionPreference = "Stop"
if (-not (Test-Path ".\.venv\Scripts\audio2score.exe")) {
    throw "audio2score is not installed in .venv. Run .\setup_windows.ps1 first."
}
$argsList = @($Song, "--meter", $Meter)
if ($AnalysisCache) {
    $argsList += @("--analysis-cache", $AnalysisCache, "--arrangement", $Arrangement)
} elseif ($Piano) {
    $argsList += @("--piano", "--arrangement", $Arrangement)
} elseif (-not $NoSeparation) {
    $argsList += "--separate-vocals"
}
if ($NoEngraving) {
    $argsList += "--no-engraving"
}
& .\.venv\Scripts\audio2score.exe @argsList
