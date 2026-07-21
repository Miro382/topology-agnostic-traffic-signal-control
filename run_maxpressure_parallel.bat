@echo off
REM ============================================================================
REM  Parallel max-pressure evaluation. Opens one window per (intersection,scale).
REM  SUMO is single-threaded, so N windows use N CPU cores. CUDA is irrelevant
REM  (no neural network). Each window writes its OWN csv, so no write conflicts:
REM      evaluation_maxpressure_<config>_s<scale>.csv
REM  Run from an Anaconda Prompt in this folder:  run_maxpressure_parallel.bat
REM  Tip: do NOT click inside the spawned windows (Windows QuickEdit pauses them).
REM  Reduce the lists below if you want to leave CPU headroom (e.g. one scale).
REM ============================================================================
cd /d "%~dp0"

for %%I in (Tory Jant Herl) do (
    for %%S in (0.2 0.6 1.0 1.4) do (
        start "MP %%I %%S" cmd /k "conda activate sumo_rl && python evaluate_maxpressure.py %%I %%S"
    )
)
echo Launched 12 max-pressure jobs in separate windows.
echo Each prints "Max-pressure evaluation complete." when its slice is done.
