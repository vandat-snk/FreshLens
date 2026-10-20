param([ValidateSet('cpu', 'cu128')][string]$Device = 'cpu')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Setup failed with exit code $LASTEXITCODE" }
}
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    Invoke-Checked -Program 'py' -Arguments @('-3.13', '-m', 'venv', '.venv')
}
Invoke-Checked -Program $projectPython -Arguments @('-m', 'pip', 'install', '--upgrade', 'pip')
Invoke-Checked -Program $projectPython -Arguments @('-m', 'pip', 'install', '-r', 'requirements.txt')
$torchRequirements = if ($Device -eq 'cpu') { 'requirements-torch-cpu.txt' } else { 'requirements-torch-cu128.txt' }
Invoke-Checked -Program $projectPython -Arguments @('-m', 'pip', 'install', '-r', $torchRequirements)
Invoke-Checked -Program $projectPython -Arguments @('-m', 'pytest', 'tests/tv3', 'tests/submission', 'tests/test_camera_dataset.py', 'tests/test_dataset_v2.py', 'tests/test_reviewed_candidate.py', '-q')
Write-Host 'Setup and runtime checks passed. Start run_app.bat.'
