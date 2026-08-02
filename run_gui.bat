@echo off
REM Launch the Derization desktop GUI on Windows.
REM Double-click this file, or run it from a terminal.
setlocal
cd /d "%~dp0"
python gui.py
if errorlevel 1 (
    echo.
    echo The GUI exited with an error. Make sure dependencies are installed:
    echo     pip install -r requirements.txt
    pause
)
endlocal
