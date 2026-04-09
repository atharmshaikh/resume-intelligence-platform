@echo off
:: train_pipeline.bat
:: -----------------
:: Windows Automation Script for Resume ML Pipeline.

:: Ensure we are in the correct directory (ml-engine)
if not exist "ml_engine" (
    echo Error: Please run this script from the ml-service/ml-engine/ directory.
    pause
    exit /b 1
)

set PYTHONPATH=%PYTHONPATH%;.
set VENV_PATH=..\..\.venv\Scripts\python.exe

echo ────────────────────────────────────────────────────────────
echo    INDUSTRIAL RESUME ML PIPELINE - AUTOMATION (WINDOWS)      
echo ────────────────────────────────────────────────────────────

:: 1. Environment Check
echo [1/4] Validating Virtual Environment...
if not exist "%VENV_PATH%" (
    echo ❌ Error: Virtual environment not found at %VENV_PATH%
    pause
    exit /b 1
)
echo      Found .venv\Scripts\python.exe

:: 2. Syncing Processed Data
echo [2/4] Syncing Processed Resumes (JSON Generation)...
"%VENV_PATH%" ml_engine/ml/inference/batch_processor.py > nul
if errorlevel 1 (
    echo ❌ Error: Batch processor failed. Check your resume files.
    pause
    exit /b 1
)

:: 3. Running Training
echo [3/4] Starting Training Pipeline (JSON-First)...
"%VENV_PATH%" ml_engine/ml/pipelines/training.py
if errorlevel 1 (
    echo ❌ Error: Training pipeline failed.
    pause
    exit /b 1
)

:: 4. Verification Pass
echo [4/4] Post-Training Verification...
"%VENV_PATH%" ml_engine/ml/inference/batch_processor.py
echo.
echo ✅ SUCCESS: Industrial Model deployed to ml_engine\ml\artifacts\
echo ────────────────────────────────────────────────────────────
pause
