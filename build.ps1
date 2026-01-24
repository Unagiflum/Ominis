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
$pyiArgs = @(
    "--noconfirm",
    "--clean",
    "--name", "Ominis",
    "--onedir",
    "--console",
    "--collect-all", "pygame",
    "--collect-binaries", "av"
)

$assetsPath = Join-Path $PSScriptRoot "assets"
if (Test-Path $assetsPath) {
    $pyiArgs += @("--add-data", "assets;assets")
} else {
    Write-Host "assets/ not found; skipping --add-data assets"
}

$settingsPath = Join-Path $PSScriptRoot "settings.json"
if (Test-Path $settingsPath) {
    $pyiArgs += @("--add-data", "settings.json;.")
} else {
    Write-Host "settings.json not found; skipping --add-data settings.json"
}

$recordingPath = Join-Path $PSScriptRoot "recording.json"
if (Test-Path $recordingPath) {
    $pyiArgs += @("--add-data", "recording.json;.")
} else {
    Write-Host "recording.json not found; skipping --add-data recording.json"
}

$pyiArgs += "main.py"

& $pythonExe -m PyInstaller @pyiArgs
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

$distDir = Join-Path $PSScriptRoot "dist\\Ominis"
if (-not (Test-Path $distDir)) { throw "Build output not found at $distDir" }

$assetsDest = Join-Path $distDir "assets"
if ((Test-Path $assetsPath) -and (-not (Test-Path $assetsDest))) {
    Write-Host "Copying assets into $assetsDest"
    Copy-Item -Path $assetsPath -Destination $assetsDest -Recurse -Force
}

$settingsDest = Join-Path $distDir "settings.json"
if ((Test-Path $settingsPath) -and (-not (Test-Path $settingsDest))) {
    Copy-Item -Path $settingsPath -Destination $settingsDest -Force
}

$recordingDest = Join-Path $distDir "recording.json"
if ((Test-Path $recordingPath) -and (-not (Test-Path $recordingDest))) {
    Copy-Item -Path $recordingPath -Destination $recordingDest -Force
}

Write-Host "Build complete: dist\\Ominis\\Ominis.exe"
