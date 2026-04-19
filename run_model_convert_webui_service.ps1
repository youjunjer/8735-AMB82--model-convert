$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$stdoutLog = Join-Path $root "model_convert_webui.out.log"
$stderrLog = Join-Path $root "model_convert_webui.err.log"

if (-not (Test-Path ".venv")) {
    python -m venv .venv
}

& .\.venv\Scripts\python.exe -m pip install -U pip | Out-Null
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt | Out-Null

while ($true) {
    Add-Content -Path $stdoutLog -Value ("[{0}] Starting model_convert_webui.py" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    & .\.venv\Scripts\python.exe model_convert_webui.py 1>> $stdoutLog 2>> $stderrLog
    Add-Content -Path $stderrLog -Value ("[{0}] model_convert_webui.py exited; restarting in 5 seconds" -f (Get-Date -Format "yyyy-MM-dd HH:mm:ss"))
    Start-Sleep -Seconds 5
}
