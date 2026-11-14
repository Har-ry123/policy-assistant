# Runs the test suite with the project's own Python.
Set-Location $PSScriptRoot
& .\.venv\Scripts\python.exe -m pytest -q
