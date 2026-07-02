@echo off
setlocal
set "PROJECT_ROOT=%~dp0"
set "STELLAR_EXE=%PROJECT_ROOT%.venv\Scripts\stellar-analyzer.exe"

if not exist "%STELLAR_EXE%" (
    echo Stellar Analyzer is not installed in .venv.
    echo Run: python -m venv .venv
    echo Then: .venv\Scripts\python.exe -m pip install -e ".[dev]"
    exit /b 1
)

"%STELLAR_EXE%" %*
exit /b %errorlevel%
