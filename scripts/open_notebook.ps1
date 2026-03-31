param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$NotebookPath
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$resolvedNotebook = Join-Path $projectRoot $NotebookPath

if (-not (Test-Path -LiteralPath $resolvedNotebook)) {
    throw "Notebook not found: $resolvedNotebook"
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonExe)) {
    throw "Project Python not found: $pythonExe"
}

$runtimeRoot = Join-Path $projectRoot ".jupyter_runtime"
$runtimeDir = Join-Path $runtimeRoot "runtime"
$stdoutLog = Join-Path $runtimeRoot "notebook-server.log"
$stderrLog = Join-Path $runtimeRoot "notebook-server.err.log"

New-Item -ItemType Directory -Force -Path $runtimeDir | Out-Null

$env:JUPYTER_CONFIG_DIR = $runtimeRoot
$env:JUPYTER_RUNTIME_DIR = $runtimeDir
$env:JUPYTER_ALLOW_INSECURE_WRITES = "true"

Get-ChildItem -LiteralPath $runtimeDir -Filter "jpserver-*-open.html" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem -LiteralPath $runtimeDir -Filter "jpserver-*.json" -ErrorAction SilentlyContinue |
    Remove-Item -Force -ErrorAction SilentlyContinue

if (Test-Path -LiteralPath $stdoutLog) {
    Remove-Item -LiteralPath $stdoutLog -Force -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $stderrLog) {
    Remove-Item -LiteralPath $stderrLog -Force -ErrorAction SilentlyContinue
}

$backgroundScript = @"
Set-Location '$projectRoot'
\$env:JUPYTER_CONFIG_DIR = '$runtimeRoot'
\$env:JUPYTER_RUNTIME_DIR = '$runtimeDir'
\$env:JUPYTER_ALLOW_INSECURE_WRITES = 'true'
& '$pythonExe' -m notebook '$NotebookPath' --no-browser 1>> '$stdoutLog' 2>> '$stderrLog'
"@

$encodedCommand = [Convert]::ToBase64String([Text.Encoding]::Unicode.GetBytes($backgroundScript))
cmd.exe /c start "" powershell.exe -NoProfile -ExecutionPolicy Bypass -EncodedCommand $encodedCommand | Out-Null

Write-Host "Starting notebook server for $NotebookPath..."

$openFile = $null
for ($attempt = 0; $attempt -lt 50; $attempt++) {
    Start-Sleep -Milliseconds 500
    $candidate = Get-ChildItem -LiteralPath $runtimeDir -Filter "jpserver-*-open.html" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
    if ($candidate -and $candidate.Length -gt 0) {
        $openFile = $candidate
        break
    }
}

if ($openFile) {
    Write-Host "Notebook server started. Opening browser..."
    try {
        Start-Process $openFile.FullName | Out-Null
    }
    catch {
        Write-Warning "Could not auto-open browser. Open this file manually: $($openFile.FullName)"
    }
    Write-Host "If the notebook page does not appear, open this file manually:"
    Write-Host "  $($openFile.FullName)"
    exit 0
}

Write-Warning "Notebook server did not finish startup in time."
if (Test-Path -LiteralPath $stdoutLog) {
    Write-Host "--- stdout tail ---"
    Get-Content -LiteralPath $stdoutLog -Tail 30 -ErrorAction SilentlyContinue
}
if (Test-Path -LiteralPath $stderrLog) {
    Write-Host "--- stderr tail ---"
    Get-Content -LiteralPath $stderrLog -Tail 30 -ErrorAction SilentlyContinue
}

exit 1
