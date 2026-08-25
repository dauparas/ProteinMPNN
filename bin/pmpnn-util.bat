@echo off

set "PYTHON_EXE=python"

set "SCRIPT_PATH=C:\ProteinMPNN\helper_scripts\parse_multiple_chains.py"

call conda activate mlfold
if errorlevel 1 (
    echo [Error] Failed to activate mlfold Environment
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_PATH%" %*
if errorlevel 1 (
    echo [Error] Something happened. Look in the error logs provided
    exit /b 1
)

call conda deactivate
if errorlevel 1 (
    echo [Error] Failed to deactivate mlfold Environment (Used by ProteinMPNN)
    exit /b 1
)
