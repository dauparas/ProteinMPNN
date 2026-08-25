@echo off
setlocal

rem Resolve the repository relative to this launcher, not the current directory.
for %%I in ("%~dp0..") do set "REPO_ROOT=%%~fI"

rem Use the active environment by default.  Set PMPNN_PYTHON to an explicit
rem interpreter path when ProteinMPNN is installed in a different environment.
if defined PMPNN_PYTHON (
    set "PYTHON_EXE=%PMPNN_PYTHON%"
) else (
    set "PYTHON_EXE=python"
)

"%PYTHON_EXE%" "%REPO_ROOT%\protein_mpnn_run.py" %*
set "EXIT_CODE=%ERRORLEVEL%"
endlocal & exit /b %EXIT_CODE%
