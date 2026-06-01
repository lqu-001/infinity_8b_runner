$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $Root

if (-not (Test-Path "third_party\geneval\prompts\evaluation_metadata.jsonl")) {
    Write-Host "GenEval prompts not found; running setup_geneval.ps1 ..."
    & "$Root\setup_geneval.ps1"
}

if (Test-Path ".venv\Scripts\python.exe") {
    $Python = ".venv\Scripts\python.exe"
} else {
    $Python = "python"
}

& $Python "$Root\geneval_run_1024.py" @args
exit $LASTEXITCODE
