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

"%PYTHON_EXE%" "%REPO_ROOT%\helper_scripts\parse_multiple_chains.py" %*
if errorlevel 1 (
    echo [Error] Something happened. Look in the error logs provided
    set "EXIT_CODE=1"
    goto :done
)

set "EXIT_CODE=0"

:done
endlocal & exit /b %EXIT_CODE%
