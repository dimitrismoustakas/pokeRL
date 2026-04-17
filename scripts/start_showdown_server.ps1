param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "pokeRL\pokemon-showdown"),
    [int]$Port = 8000,
    [switch]$InstallOnly,
    [switch]$Update,
    [switch]$Detached,
    [int]$ReadyTimeoutSec = 15
)

$ErrorActionPreference = "Stop"

if ($ReadyTimeoutSec -lt 1) {
    throw "-ReadyTimeoutSec must be at least 1 second."
}

function Test-ShowdownPort {
    param(
        [int]$Port
    )

    $client = [System.Net.Sockets.TcpClient]::new()
    try {
        $async = $client.BeginConnect("127.0.0.1", $Port, $null, $null)
        $connected = $async.AsyncWaitHandle.WaitOne(500)
        return [bool]($connected -and $client.Connected)
    }
    catch {
        return $false
    }
    finally {
        $client.Close()
    }
}

function Wait-ShowdownReady {
    param(
        [int]$Port,
        [int]$TimeoutSec,
        [int[]]$CleanupProcessIds = @()
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSec)
    while ((Get-Date) -lt $deadline) {
        if (Test-ShowdownPort -Port $Port) {
            return
        }
        Start-Sleep -Milliseconds 250
    }

    foreach ($procId in $CleanupProcessIds) {
        if (Get-Process -Id $procId -ErrorAction SilentlyContinue) {
            Stop-Process -Id $procId -Force -ErrorAction SilentlyContinue
        }
    }

    throw "Pokemon Showdown did not become reachable on localhost:$Port within $TimeoutSec seconds."
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

if (Test-ShowdownPort -Port $Port) {
    Write-Output "Pokemon Showdown is already running on localhost:$Port"
    exit 0
}

if ($Detached) {
    $startProcessArgs = @{
        FilePath = "node"
        ArgumentList = @("pokemon-showdown", "start", "--no-security", $Port.ToString())
        WorkingDirectory = $InstallDir
        WindowStyle = "Hidden"
        PassThru = $true
    }
    $process = Start-Process @startProcessArgs
    Wait-ShowdownReady -Port $Port -TimeoutSec $ReadyTimeoutSec -CleanupProcessIds @($process.Id)
    Write-Output "Pokemon Showdown started on localhost:$Port (pid $($process.Id))"
    exit 0
}

node pokemon-showdown start --no-security $Port