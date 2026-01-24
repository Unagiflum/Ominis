$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

$venvPath = Join-Path $PSScriptRoot ".venv"
$pythonExe = Join-Path $venvPath "Scripts\\python.exe"

if (-not (Test-Path $pythonExe)) {
    Write-Host "Creating virtual environment at $venvPath"
    python -m venv $venvPath
}

Write-Host "Installing requirements"
& $pythonExe -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { throw "pip upgrade failed with exit code $LASTEXITCODE" }
& $pythonExe -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed with exit code $LASTEXITCODE" }

Write-Host "Building Ominis (console, full functionality)"
& $pythonExe -m PyInstaller --noconfirm --clean --name Ominis --onedir --console `
    --add-data "assets;assets" `
    --add-data "settings.json;." `
    --add-data "recording.json;." `
    --collect-all pygame `
    --collect-binaries av `
    main.py
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$distDir = Join-Path $PSScriptRoot "dist\\Ominis"
if (-not (Test-Path $distDir)) { throw "Build output not found at $distDir" }

$assetsDest = Join-Path $distDir "assets"
if (-not (Test-Path $assetsDest)) {
    Write-Host "Copying assets into $assetsDest"
    Copy-Item -Path (Join-Path $PSScriptRoot "assets") -Destination $assetsDest -Recurse -Force
}

$settingsDest = Join-Path $distDir "settings.json"
if (-not (Test-Path $settingsDest)) {
    Copy-Item -Path (Join-Path $PSScriptRoot "settings.json") -Destination $settingsDest -Force
}

$recordingDest = Join-Path $distDir "recording.json"
if (-not (Test-Path $recordingDest)) {
    Copy-Item -Path (Join-Path $PSScriptRoot "recording.json") -Destination $recordingDest -Force
}

Write-Host "Build complete: dist\\Ominis\\Ominis.exe"
