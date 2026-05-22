$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdParty = Join-Path $Root "third_party"
$Infinity = Join-Path $ThirdParty "Infinity"

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (-not (Test-Path $Infinity)) {
    git clone https://github.com/FoundationVision/Infinity.git $Infinity
} else {
    Write-Host "Official Infinity repo already exists: $Infinity"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Install official dependencies from third_party/Infinity/requirements.txt"
Write-Host "  2. Run: python download_weights.py --resolution 1024"
Write-Host "  3. Run: python generate_infinity_8b.py --prompt `"alien spaceship enterprise`""
