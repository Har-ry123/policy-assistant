# Starts the API with the project's own Python, so you never need to activate the environment.
#   .\run_api.ps1            (add --reload by editing the line below if you want auto-restart)
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m uvicorn app.api:app_factory --factory --port 8000
