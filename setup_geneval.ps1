$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$ThirdParty = Join-Path $Root "third_party"
$GenEval = Join-Path $ThirdParty "geneval"

New-Item -ItemType Directory -Force -Path $ThirdParty | Out-Null

if (-not (Test-Path $GenEval)) {
    git clone https://github.com/djghosh13/geneval.git $GenEval
} else {
    Write-Host "GenEval repo already exists: $GenEval"
}

$Metadata = Join-Path $GenEval "prompts\evaluation_metadata.jsonl"
if (Test-Path $Metadata) {
    $lines = (Get-Content $Metadata | Measure-Object -Line).Lines
    Write-Host "GenEval prompts: $lines lines in evaluation_metadata.jsonl"
}

Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Generate: python geneval_generate.py --outdir outputs\geneval --resolution 1024"
Write-Host "  2. Install GenEval eval deps (Linux/WSL recommended): see README GenEval section"
Write-Host "  3. Download detector: python geneval_evaluate.py outputs\geneval --outfile outputs\geneval\results.jsonl"
Write-Host "     (first run downloads Mask2Former weights on Windows; Linux can use evaluation/download_models.sh)"
