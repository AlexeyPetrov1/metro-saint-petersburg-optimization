param([int]$Port = 8501)
$ErrorActionPreference = 'Stop'
$uiRoot = $PSScriptRoot
$uiPython = Join-Path $uiRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $uiPython)) {
    throw 'Создайте окружение интерфейса по инструкции README.md.'
}
Push-Location -LiteralPath $uiRoot
try {
    & $uiPython -m streamlit run app.py --server.address 127.0.0.1 --server.port $Port --server.headless true --browser.gatherUsageStats false
} finally {
    Pop-Location
}
