@echo OFF
setlocal EnableDelayedExpansion

:: Set the path to your local Anaconda installation
set ANACONDA_PATH=%USERPROFILE%\Anaconda3

:: Set the name of your Conda environment
set CONDA_ENV=ima2025

:: Activate the Conda environment
call %ANACONDA_PATH%\Scripts\activate.bat %CONDA_ENV%

:: Run Python in the activated environment
python motion_clustering.py

:: Deactivate the environment
call conda deactivate

:: Keep the command window open
cmd /k