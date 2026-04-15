param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "pokeRL\pokemon-showdown"),
    [switch]$InstallOnly,
    [switch]$Update
)

$ErrorActionPreference = "Stop"

function Test-ShowdownPort {
    $result = Test-NetConnection -ComputerName localhost -Port 8000 -WarningAction SilentlyContinue
    return [bool]$result.TcpTestSucceeded
}

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw "git is required to set up Pokemon Showdown."
}

if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    throw "node is required to run Pokemon Showdown."
}

if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    throw "npm is required to install Pokemon Showdown dependencies."
}

if (-not (Test-Path $InstallDir)) {
    New-Item -ItemType Directory -Path (Split-Path $InstallDir -Parent) -Force | Out-Null
    git clone --depth 1 https://github.com/smogon/pokemon-showdown.git "$InstallDir"
}

Set-Location "$InstallDir"

if ($Update) {
    git pull --ff-only
}

if (-not (Test-Path (Join-Path $InstallDir "node_modules")) -or $Update) {
    npm install
}

if ($InstallOnly) {
    Write-Output "Pokemon Showdown is installed at $InstallDir"
    exit 0
}

if (Test-ShowdownPort) {
    Write-Output "Pokemon Showdown is already running on localhost:8000"
    exit 0
}

node pokemon-showdown start --no-security