@echo off
setlocal

rem Launch ProteinMPNN Studio with the same no-setup Conda workflow as pmpnn.
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

if defined PMPNN_PYTHON (
    "%PMPNN_PYTHON%" "%REPO_ROOT%\protein_mpnn_gui.py"
    set "EXIT_CODE=%ERRORLEVEL%"
    endlocal & exit /b %EXIT_CODE%
)

if not defined PMPNN_ENV set "PMPNN_ENV=proteinmpnn"
where conda >nul 2>nul
if errorlevel 1 (
    echo [Error] Conda is required for automatic setup but was not found on PATH.
    echo Install Miniconda or Anaconda, then launch ProteinMPNN Studio again.
    pause
    endlocal & exit /b 1
)

conda env list | findstr /R /C:"^%PMPNN_ENV%[ *]" >nul
if errorlevel 1 (
    echo [ProteinMPNN] Creating Conda environment '%PMPNN_ENV%'...
    call conda create --yes --name "%PMPNN_ENV%" python=3.10 pip
    if errorlevel 1 ( endlocal & exit /b 1 )
)

call conda run --no-capture-output --name "%PMPNN_ENV%" python -c "import tkinter" >nul 2>nul
if errorlevel 1 (
    echo [ProteinMPNN] Installing the desktop GUI component in '%PMPNN_ENV%'...
    call conda install --yes --name "%PMPNN_ENV%" tk
    if errorlevel 1 ( endlocal & exit /b 1 )
)

call conda run --no-capture-output --name "%PMPNN_ENV%" python -c "import numpy, torch" >nul 2>nul
if errorlevel 1 (
    echo [ProteinMPNN] Installing ProteinMPNN requirements in '%PMPNN_ENV%'...
    call conda run --no-capture-output --name "%PMPNN_ENV%" python -m pip install --requirement "%REPO_ROOT%\requirements.txt"
    if errorlevel 1 ( endlocal & exit /b 1 )
)

call conda run --no-capture-output --name "%PMPNN_ENV%" python "%REPO_ROOT%\protein_mpnn_gui.py"
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
