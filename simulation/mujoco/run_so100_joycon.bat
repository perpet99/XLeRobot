@echo off
rem ---------------------------------------------------------------------------
rem Launch the SO100 MuJoCo Joy-Con teleoperation in the "xlerobot" conda env.
rem
rem   run_so100_joycon.bat                 Joy-Con if present, else keyboard
rem   run_so100_joycon.bat --device joycon require a Joy-Con
rem   run_so100_joycon.bat --selftest      headless IK/FK check, no viewer
rem
rem Any argument is passed straight through to so100_joycon_mujoco.py.
rem Override the defaults from the shell if your setup differs, e.g.
rem   set CONDA_ROOT=C:\Users\me\miniconda3
rem   set ENV_NAME=my_env
rem ---------------------------------------------------------------------------
setlocal

set "SCRIPT_DIR=%~dp0"
if not defined ENV_NAME set "ENV_NAME=xlerobot"

rem --- locate the conda installation ------------------------------------------
if defined CONDA_ROOT goto :have_root

for %%R in (
  "%USERPROFILE%\miniconda3"
  "%USERPROFILE%\anaconda3"
  "%LOCALAPPDATA%\miniconda3"
  "%LOCALAPPDATA%\Continuum\anaconda3"
  "C:\ProgramData\miniconda3"
  "C:\ProgramData\Anaconda3"
  "D:\Anaconda3"
  "D:\miniconda3"
) do if exist "%%~R\Scripts\activate.bat" (
  set "CONDA_ROOT=%%~R"
  goto :have_root
)

echo [error] No conda installation found.
echo         Set CONDA_ROOT to it and retry, for example:
echo             set CONDA_ROOT=D:\Anaconda3
echo.
pause
exit /b 1

:have_root
echo [info] conda root : %CONDA_ROOT%
echo [info] environment: %ENV_NAME%

rem --- activate the environment -----------------------------------------------
call "%CONDA_ROOT%\Scripts\activate.bat" "%ENV_NAME%"
if errorlevel 1 (
  echo.
  echo [error] Could not activate the "%ENV_NAME%" environment.
  echo         Create it first:
  echo             conda create -y -n %ENV_NAME% python=3.10
  echo             conda activate %ENV_NAME%
  echo             pip install -r "%SCRIPT_DIR%requirements_so100.txt"
  echo.
  pause
  exit /b 1
)

rem --- run ---------------------------------------------------------------------
python "%SCRIPT_DIR%so100_joycon_mujoco.py" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
  echo.
  echo [error] so100_joycon_mujoco.py exited with code %RC%.
  pause
)

exit /b %RC%
