@echo off
echo Minimal Setup: Creating venv and installing requirements.
echo Assumes Python 3 is correctly installed and in PATH.
echo.

REM Define virtual environment directory relative to this script
set "VENV_DIR=%~dp0env"
REM Define requirements file relative to this script
set "REQUIREMENTS_FILE=%~dp0requirements.txt"

REM Create virtual environment
echo Creating virtual environment in "%VENV_DIR%"...
python -m venv "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment.
    echo Make sure Python 3 is installed and 'python -m venv' works.
    pause
    exit /b 1
)
echo Virtual environment created (or already existed and was updated).
echo.

REM Activate virtual environment and install requirements
echo Activating virtual environment and installing requirements...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)

echo Installing from "%REQUIREMENTS_FILE%"...
pip install -r "%REQUIREMENTS_FILE%"
if %errorlevel% neq 0 (
    echo ERROR: Failed to install requirements.
    echo Make sure pip is working and requirements.txt is correct.
    pause
    exit /b 1
)

echo.
echo ====================================================================
echo Minimal setup complete.
echo Virtual environment 'env' should be ready with requirements installed.
echo ====================================================================
echo.
pause