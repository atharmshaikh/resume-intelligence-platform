@echo off
:: =============================================================================
:: run_test.bat  –  One-command test runner for ml-engine (Windows Native)
:: 
:: Usage (from any directory inside the project):
::   run_test.bat
:: =============================================================================

setlocal enabledelayedexpansion

:: ── Locate the project root (folder containing .venv) ────────────────────────
set SCRIPT_DIR=%~dp0
set SCRIPT_DIR=%SCRIPT_DIR:~0,-1%
set PROJECT_ROOT=%SCRIPT_DIR%\..\..
set VENV_PYTHON=%PROJECT_ROOT%\.venv\Scripts\python.exe

:: ── Sanity checks ────────────────────────────────────────────────────────────
if not exist "%VENV_PYTHON%" (
    echo [ERROR] CRITICAL: Virtual Environment not found at: %PROJECT_ROOT%\.venv
    echo         Security requires execution isolated within a .venv context.
    echo         Run:  python -m venv .venv ^&^& .venv\Scripts\pip install -e ml-service\ml-engine
    exit /b 1
)

:: ── Require Python 3.12+ ─────────────────────────────────────────────────────
for /f "delims=" %%I in ('"%VENV_PYTHON%" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"') do set PYTHON_VERSION=%%I

:: Simple string comparison (Works safely if format is 3.12, 3.13, etc vs 3.10 and 3.11)
if "%PYTHON_VERSION%"=="3.8" goto :version_error
if "%PYTHON_VERSION%"=="3.9" goto :version_error
if "%PYTHON_VERSION%"=="3.10" goto :version_error
if "%PYTHON_VERSION%"=="3.11" goto :version_error
goto :version_ok

:version_error
echo [ERROR] Security/Stability Error: Python %PYTHON_VERSION% detected.
echo         This project strictly forces Python 3.12+ to prevent unknown bugs and vulnerabilities.
exit /b 1

:version_ok

echo.
echo ===================================================================
echo   [TEST]  Resume ML-Engine Test Suite
echo   Python : %VENV_PYTHON%
echo   Engine : %SCRIPT_DIR%
echo ===================================================================
echo.

cd /d "%SCRIPT_DIR%"

:: Clear bytecode cache to avoid stale-module bugs (Industry Standard)
del /S /Q __pycache__\* >nul 2>&1
for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d" >nul 2>&1

:: Run the pipeline test securely
:: (Note: Windows CMD lacks native timeout/ulimit without PowerShell wrappers, 
:: but we catch standard execution failures cleanly)
"%VENV_PYTHON%" tests\test_pipeline.py

set EXIT_CODE=%ERRORLEVEL%

if %EXIT_CODE% EQU 0 (
    echo.
    echo ===================================================================
    echo   [PASS]  All tests passed successfully.
    echo ===================================================================
) else (
    echo.
    echo ===================================================================
    echo   [FAIL]  Tests FAILED (exit code %EXIT_CODE%)
    echo ===================================================================
)

exit /b %EXIT_CODE%
