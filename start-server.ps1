$ErrorActionPreference = "Stop"

$port = 8080
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$pythonCandidates = @(
  "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe",
  "python",
  "py"
)

$python = $null
foreach ($candidate in $pythonCandidates) {
  try {
    $cmd = Get-Command $candidate -ErrorAction Stop
    $python = $cmd.Source
    break
  } catch {
    if (Test-Path -LiteralPath $candidate) {
      $python = $candidate
      break
    }
  }
}

if (-not $python) {
  Write-Host "Python was not found. Install Python or run this from Codex where the bundled Python exists." -ForegroundColor Red
  exit 1
}

Write-Host "Serving:" $root -ForegroundColor Cyan
Write-Host "URL: http://127.0.0.1:$port/index.html" -ForegroundColor Green
Write-Host "Press Ctrl+C to stop." -ForegroundColor Yellow
Set-Location -LiteralPath $root
& $python -u -m http.server $port --bind 127.0.0.1
