@echo off
setlocal

REM Define paths relative to this script
set SCRIPT_DIR=%~dp0
set VENV_DIR=%SCRIPT_DIR%env
set PYTHON_SCRIPT=%SCRIPT_DIR%src\sbs_to_depth.py
set INPUT_DIR_RELATIVE=input_images
set OUTPUT_DIR_RELATIVE=output_images

set INPUT_DIR=%SCRIPT_DIR%%INPUT_DIR_RELATIVE%
set OUTPUT_DIR=%SCRIPT_DIR%%OUTPUT_DIR_RELATIVE%


REM Create directories if they don't exist
if not exist "%INPUT_DIR%" mkdir "%INPUT_DIR%"
if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

REM Check if virtual environment exists
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found in %VENV_DIR%.
    echo Please run setup_windows.bat first to create it.
    pause
    exit /b 1
)

REM Activate virtual environment
call "%VENV_DIR%\Scripts\activate.bat"

echo.
echo --- Stereo Depth Map Generator ---
echo.
echo Make sure your SBS image is in the '%INPUT_DIR_RELATIVE%' folder.
echo Output will be saved in the '%OUTPUT_DIR_RELATIVE%' folder.
echo.

:askimage
set /p IMAGE_FILENAME="Enter the filename of your SBS image (e.g., sbs_image.png): "

if "%IMAGE_FILENAME%"=="" (
    echo No filename entered. Please try again.
    goto askimage
)

set INPUT_IMAGE_PATH="%INPUT_DIR%\%IMAGE_FILENAME%"
REM For the python script, we can pass the path relative to the script or an absolute path.
REM Python script will handle it. The argparse will take the string.
set PYTHON_IMAGE_ARG="%INPUT_DIR_RELATIVE%\%IMAGE_FILENAME%" 
REM Check if the actual file exists
if not exist %INPUT_IMAGE_PATH% (
    echo ERROR: Image %INPUT_IMAGE_PATH% not found.
    echo Please make sure the file exists in the '%INPUT_DIR_RELATIVE%' folder and the name is correct.
    goto askimage
)

echo.
echo Optional: To calculate actual depth, provide focal length (pixels) and baseline (meters).
set /p FOCAL_LENGTH_ARG="Enter focal length in pixels (e.g., 700) or press Enter to skip depth calculation: "
set /p BASELINE_ARG="Enter baseline in meters (e.g., 0.12) or press Enter to skip depth calculation: "

set EXTRA_ARGS=
if not "%FOCAL_LENGTH_ARG%"=="" if not "%BASELINE_ARG%"=="" (
    set EXTRA_ARGS=--focal_length %FOCAL_LENGTH_ARG% --baseline %BASELINE_ARG%
)

echo.
echo Running stereo depth processing for %PYTHON_IMAGE_ARG%...
echo.

REM Run the Python script
REM Pass input image path and output directory as arguments
python "%PYTHON_SCRIPT%" %PYTHON_IMAGE_ARG% --output_dir "%OUTPUT_DIR_RELATIVE%" %EXTRA_ARGS%

if %errorlevel% neq 0 (
    echo An error occurred while running the Python script.
) else (
    echo Processing complete. Check the '%OUTPUT_DIR_RELATIVE%' folder for results.
)

echo.
echo Press any key to close this window.
pause >nul
endlocal