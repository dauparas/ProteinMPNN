@echo off

set "PYTHON_EXE=python"

set "SCRIPT_PATH=C:\ProteinMPNN\protein_mpnn_run.py"

call conda activate mlfold
if errorlevel 1 (
    echo [Error] Failed to activate mlfold Environment. It is required for ProteinMPNN :(
    exit /b 1
)

"%PYTHON_EXE%" "%SCRIPT_PATH%" %*

call conda deactivate
if errorlevel 1 (
    echo [Error] Failed to deactivate mlfold Environment (Used by ProteinMPNN)
    exit /b 1
)
