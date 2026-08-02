@echo off
REM ===================================================================
REM  Derization - one-click launcher
REM  Double-click this file. On first run it builds its own isolated
REM  Python environment (.venv) and installs everything it needs; after
REM  that it just launches the app.
REM ===================================================================
setlocal enableextensions enabledelayedexpansion
title Derization - Speaker Separation
cd /d "%~dp0"

echo.
echo ============================================
echo    Derization - Speaker Separation
echo ============================================
echo.

REM --- 1. Locate a Python 3 interpreter -------------------------------
set "PY="
where py        >nul 2>&1 && set "PY=py -3"
if not defined PY where python >nul 2>&1 && set "PY=python"
if not defined PY (
    echo [ERROR] Python 3 was not found on this computer.
    echo.
    echo Install Python 3.10+ from https://www.python.org/downloads/
    echo During install, tick "Add python.exe to PATH", then run this again.
    echo.
    pause
    exit /b 1
)

REM --- 2. Create the local virtual environment ------------------------
set "VENV=%~dp0.venv"
set "VPY=%VENV%\Scripts\python.exe"
if not exist "%VPY%" (
    echo Creating a self-contained environment in ".venv" ...
    %PY% -m venv "%VENV%"
    if errorlevel 1 (
        echo [ERROR] Could not create the virtual environment.
        pause
        exit /b 1
    )
)

REM --- 3. First-run dependency setup (once) ---------------------------
set "MARKER=%VENV%\.derization_ready"
if not exist "%MARKER%" (
    echo.
    echo ------------------------------------------------------------
    echo   FIRST-TIME SETUP - installing dependencies into .venv
    echo   Live download progress bars from pip appear below.
    echo   ^(This happens only once; later launches are instant.^)
    echo ------------------------------------------------------------
    echo.

    echo [1/5] Updating the package installer ^(pip^)...
    "%VPY%" -m pip install --upgrade pip --progress-bar on
    if errorlevel 1 goto setup_failed

    set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
    set "TORCH_KIND=CPU"
    where nvidia-smi >nul 2>&1
    if not errorlevel 1 (
        set "TORCH_INDEX=https://download.pytorch.org/whl/cu121"
        set "TORCH_KIND=NVIDIA GPU (CUDA)"
    )

    echo.
    echo [2/5] Installing the RVC/UVR vocal-isolation engine ^(audio-separator^)...
    "%VPY%" -m pip install --progress-bar on "audio-separator[gpu]"
    if errorlevel 1 goto setup_failed

    echo.
    echo [3/5] Installing audio + ML libraries ^(numpy, librosa, noisereduce, ...^)...
    "%VPY%" -m pip install --progress-bar on -r "%~dp0requirements.txt"
    if errorlevel 1 goto setup_failed

    echo.
    echo [4/5] Installing speaker + voice-activity models ^(speechbrain, silero-vad^)...
    "%VPY%" -m pip install --progress-bar on speechbrain silero-vad
    if errorlevel 1 goto setup_failed

    echo.
    echo [5/5] Installing PyTorch for !TORCH_KIND! - the largest step, done LAST
    echo       so the GPU build isn't overwritten by another package.
    echo       Download is ~200 MB ^(CPU^) or ~2.4 GB ^(GPU^); pip shows a live
    echo       progress bar, then it unpacks quietly for 1-3 minutes - please wait.
    echo.
    "%VPY%" -m pip install --progress-bar on --force-reinstall torch torchaudio --index-url !TORCH_INDEX!
    if errorlevel 1 goto setup_failed

    echo ready> "%MARKER%"
    echo.
    echo ============================================================
    echo   Setup complete - all dependencies installed in .venv
    echo ============================================================
    echo.
)

REM --- 4. Launch the GUI ---------------------------------------------
echo Launching Derization ...
"%VPY%" "%~dp0gui.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
    echo.
    echo The app exited with code %RC%.
    pause
)
exit /b %RC%

:setup_failed
echo.
echo [ERROR] Dependency installation failed. Check your internet connection
echo and run this file again. To install manually:
echo     "%VPY%" -m pip install -r requirements.txt
echo.
pause
exit /b 1
