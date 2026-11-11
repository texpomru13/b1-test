# Windows wrapper (DEPLOYMENT_SPEC §3): runs ./run inside a PREPARED WSL2 distro
# with Docker integration. It never installs WSL, never bypasses the PowerShell
# execution policy, and preserves the exit code. Cyrillic/space paths supported.
# STATUS: written per spec; live Windows smoke is `not_run` (no Windows machine
# in this project's environment) — do not claim native Windows inference.
param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Args)

$ErrorActionPreference = "Stop"

function Fail($msg, $code) {
    Write-Error $msg
    exit $code
}

if (-not (Get-Command wsl.exe -ErrorAction SilentlyContinue)) {
    Fail "WSL2 is required. Install it manually (wsl --install), then enable Docker Desktop WSL integration." 10
}

$distros = (wsl.exe -l -q) -split "`n" | Where-Object { $_ -ne "" }
if (-not $distros) {
    Fail "No WSL distro found. Install one (e.g. Ubuntu) and enable Docker Desktop WSL integration." 10
}

# Translate this script's directory to a WSL path (handles spaces/Cyrillic).
$winPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$wslPath = (wsl.exe wslpath -a "$winPath").Trim()
if (-not $wslPath) { Fail "Could not translate '$winPath' into a WSL path." 11 }

wsl.exe --cd "$wslPath" sh ./run @Args
exit $LASTEXITCODE
