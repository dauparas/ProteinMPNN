@echo off
setlocal

rem Generate the local icon and add launch shortcuts for the current user.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0install-pmpnn-shortcuts.ps1"
set "EXIT_CODE=%ERRORLEVEL%"
if not "%EXIT_CODE%"=="0" pause
endlocal & exit /b %EXIT_CODE%
