$ErrorActionPreference = "Stop"

$repo = "https://github.com/divisonofficer/Ambient-robust-Inverse-Rendering-using-Active-RGB-NIR-Imaging.git"
$git = Get-Command git -ErrorAction SilentlyContinue

if (-not $git) {
  Write-Host "Git was not found on PATH. Install Git for Windows, then rerun this script." -ForegroundColor Red
  exit 1
}

git branch -M main
git config gc.auto 0
git config maintenance.auto false
git config core.fsmonitor false
git remote remove origin 2>$null
git remote add origin $repo
git add .

$status = git status --porcelain
if ($status) {
  git commit -m "Deploy RGB-NIR project page"
} else {
  Write-Host "No local changes to commit." -ForegroundColor Yellow
}

git push -u origin main
