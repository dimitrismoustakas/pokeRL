param(
    [string]$InstallDir = (Join-Path $env:LOCALAPPDATA "pokeRL\pokemon-showdown"),
    [int]$Port = 8000,
    [switch]$InstallOnly,
    [switch]$Update,
    [switch]$Detached,
    [int]$ReadyTimeoutSec = 60,
    [int]$NetworkProcesses = 1,
    [int]$SimulatorProcesses = 1,
    [int]$ValidatorProcesses = 1,
    [switch]$EnableProcessDebugging
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

function Set-ShowdownTrainingConfig {
    param(
        [string]$InstallDir,
        [int]$NetworkProcesses,
        [int]$SimulatorProcesses,
        [int]$ValidatorProcesses,
        [bool]$EnableProcessDebugging
    )

    $configDir = Join-Path $InstallDir "config"
    $configPath = Join-Path $configDir "config.js"
    $backupPath = Join-Path $configDir "config.js.pokerl.bak"

    if (-not (Test-Path $configPath)) {
        throw "Could not find config.js under $configDir."
    }
    if (-not (Test-Path $backupPath)) {
        Copy-Item -LiteralPath $configPath -Destination $backupPath
    }

    $content = Get-Content -Path $backupPath -Raw
    $debugLiteral = if ($EnableProcessDebugging) { "true" } else { "false" }
    $replacements = @(
        @{ Pattern = '(?m)^\s*network:\s*\d+,'; Replacement = "`tnetwork: $NetworkProcesses,"; Label = "network worker count" },
        @{ Pattern = '(?m)^\s*simulator:\s*\d+,'; Replacement = "`tsimulator: $SimulatorProcesses,"; Label = "simulator worker count" },
        @{ Pattern = '(?m)^\s*validator:\s*\d+,'; Replacement = "`tvalidator: $ValidatorProcesses,"; Label = "validator worker count" },
        @{ Pattern = '(?m)^exports\.debugsimprocesses = (true|false);'; Replacement = "exports.debugsimprocesses = $debugLiteral;"; Label = "simulator debug flag" },
        @{ Pattern = '(?m)^exports\.debugvalidatorprocesses = (true|false);'; Replacement = "exports.debugvalidatorprocesses = $debugLiteral;"; Label = "validator debug flag" },
        @{ Pattern = '(?m)^exports\.debugdexsearchprocesses = (true|false);'; Replacement = "exports.debugdexsearchprocesses = $debugLiteral;"; Label = "dexsearch debug flag" }
    )

    foreach ($replacement in $replacements) {
        $match = [regex]::Match($content, $replacement.Pattern)
        if (-not $match.Success) {
            throw "Could not update $($replacement.Label) in $configPath."
        }
        $content = [regex]::Replace($content, $replacement.Pattern, $replacement.Replacement, 1)
    }

    Set-Content -Path $configPath -Value $content -NoNewline
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

Set-ShowdownTrainingConfig `
    -InstallDir $InstallDir `
    -NetworkProcesses $NetworkProcesses `
    -SimulatorProcesses $SimulatorProcesses `
    -ValidatorProcesses $ValidatorProcesses `
    -EnableProcessDebugging:$EnableProcessDebugging

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
