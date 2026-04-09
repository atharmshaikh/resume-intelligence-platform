@echo off
:: =============================================================================
:: Resume ML Engine - Batch Run Script (Windows)
:: =============================================================================

set VENV_DIR=..\..\.venv

<<<<<<< HEAD
=======
:: 0. Ensure we are in the correct directory
cd /d "%~dp0"

>>>>>>> feature/optimization-and-refactor
:: 1. Check if venv exists, create if missing
if not exist "%VENV_DIR%" (
    echo 🏗️  Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
    
    echo 📦  Installing dependencies from pyproject.toml...
    %VENV_DIR%\Scripts\pip install --upgrade pip
    %VENV_DIR%\Scripts\pip install -e .
)

:: 2. Activate and Run
echo 🚀  Running Resume ML Batch Processor...
%VENV_DIR%\Scripts\python -m ml_engine.ml.inference.batch_processor %*
