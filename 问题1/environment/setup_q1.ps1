param(
    [switch]$SkipMain,
    [switch]$SkipMfa,
    [switch]$SkipOpenFace,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$mainPrefix = Join-Path $projectRoot ".envs\q1"
$mfaPrefix = Join-Path $projectRoot ".envs\mfa"
$mainSpec = Join-Path $PSScriptRoot "environment-q1.yml"
$gpuSpec = Join-Path $PSScriptRoot "requirements-q1-gpu-cu130.txt"
$mfaSpec = Join-Path $PSScriptRoot "environment-mfa.yml"
$toolsDirName = ([char]0x5DE5) + ([char]0x5177)
$toolsRoot = Join-Path $projectRoot $toolsDirName
$openFaceRoot = Join-Path $toolsRoot "OpenFace_2.2.0_win_x64"
$openFaceExe = Join-Path $openFaceRoot "FeatureExtraction.exe"

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was not found. Install Miniconda or Anaconda first."
}

New-Item -ItemType Directory -Force -Path (Join-Path $projectRoot ".envs") | Out-Null
New-Item -ItemType Directory -Force -Path $toolsRoot | Out-Null

if (-not $SkipMain) {
    if (Test-Path (Join-Path $mainPrefix "python.exe")) {
        conda env update --prefix $mainPrefix --file $mainSpec
    } else {
        conda env create --prefix $mainPrefix --file $mainSpec
    }
    if ($CpuOnly) {
        & (Join-Path $mainPrefix "python.exe") -m pip install "torch==2.14.0"
    } else {
        & (Join-Path $mainPrefix "python.exe") -m pip install --requirement $gpuSpec
    }
}

if (-not $SkipMfa) {
    if (Test-Path (Join-Path $mfaPrefix "python.exe")) {
        conda env update --prefix $mfaPrefix --file $mfaSpec
    } else {
        conda env create --prefix $mfaPrefix --file $mfaSpec
    }
}

if (-not $SkipOpenFace -and -not (Test-Path $openFaceExe)) {
    $archive = Join-Path $env:TEMP "OpenFace_2.2.0_win_x64.zip"
    $extractRoot = Join-Path $env:TEMP ("OpenFace_v2.2.0_win_x64_" + [guid]::NewGuid().ToString("N"))
    $downloadUrl = "https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip"

    Invoke-WebRequest -Uri $downloadUrl -OutFile $archive
    Expand-Archive -LiteralPath $archive -DestinationPath $extractRoot

    $candidate = Get-ChildItem -Path $extractRoot -Filter "FeatureExtraction.exe" -Recurse | Select-Object -First 1
    if (-not $candidate) {
        throw "FeatureExtraction.exe was not found in the OpenFace archive."
    }
    $sourceRoot = $candidate.Directory.FullName
    $sourceFull = [System.IO.Path]::GetFullPath($sourceRoot)
    $tempFull = [System.IO.Path]::GetFullPath($env:TEMP).TrimEnd('\') + '\'
    $destinationFull = [System.IO.Path]::GetFullPath($openFaceRoot)
    $toolsFull = [System.IO.Path]::GetFullPath($toolsRoot).TrimEnd('\') + '\'
    if (-not $sourceFull.StartsWith($tempFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The OpenFace source path is outside the temporary directory."
    }
    if (-not $destinationFull.StartsWith($toolsFull, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "The OpenFace destination is outside the project tools directory."
    }
    Move-Item -LiteralPath $sourceRoot -Destination $openFaceRoot

    $modelDownloader = Join-Path $openFaceRoot "download_models.ps1"
    if (Test-Path $modelDownloader) {
        Push-Location $openFaceRoot
        try {
            & $modelDownloader
        } finally {
            Pop-Location
        }
    }
}

& (Join-Path $mainPrefix "python.exe") (Join-Path $PSScriptRoot "check_q1_environment.py")
