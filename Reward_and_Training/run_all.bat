@echo off
REM ==========================================================================
REM  run_all.bat - every experiment for the "depth" part of the report.
REM  Run it from the Simulation folder with the .venv active:
REM      .venv\Scripts\activate
REM      run_all.bat
REM  It can be stopped and restarted at any time: finished trainings are skipped
REM  (--skip-if-done). Results go to results\depth\ . Budget: 11 trainings of
REM  300k steps + the analyses; roughly one night on a typical laptop.
REM ==========================================================================
set PYTHONUTF8=1
set STEPS=300000
set COMMON=--timesteps %STEPS% --eval-freq 50000 --eval-episodes 10 --skip-if-done

echo.
echo === 0. quick checks ======================================================
python -m pytest -q test_flip_policy.py test_baselines.py || goto :fail

echo.
echo === 1. final configuration, 3 seeds ======================================
python train_ppo.py %COMMON% --seed 0 --tag final || goto :fail
python train_ppo.py %COMMON% --seed 1 --tag final || goto :fail
python train_ppo.py %COMMON% --seed 2 --tag final || goto :fail

echo.
echo === 2. ablations: remove ONE component at a time ==========================
python train_ppo.py --timesteps 0 --seed 0 --tag bc_only --skip-if-done || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag no_bc --bc-episodes 0 --lr 3e-4 --log-std-init -1.0 --ent-start 0.005 --demo-prob 0.5 || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag symmetric --symmetric || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag no_prog --w-prog 0 || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag no_demo --demo-prob 0 || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag motors --action-mode motors --bc-episodes 0 --lr 3e-4 --log-std-init -1.0 --ent-start 0.005 --demo-prob 0.5 || goto :fail

echo.
echo === 5. memory experiment: does an integral input fix the wind drift? =====
python train_ppo.py %COMMON% --seed 0 --tag wind_ctrl --wind-prob 0.7 --wind-max 5 --lam-pos 0.05 --eval-wind 5 || goto :fail
python train_ppo.py %COMMON% --seed 0 --tag wind_integral --pos-integral --wind-prob 0.7 --wind-max 5 --lam-pos 0.05 --eval-wind 5 || goto :fail

echo.
echo === aggregate: tables, learning curves, ablation bars =====================
python aggregate_runs.py --runs runs --episodes 20 || goto :fail

echo.
echo === 3. robustness curves (delivered model) ================================
python robustness_sweep.py --model models/ppo_flip.zip --episodes 10 || goto :fail

echo.
echo === 4. learned flip vs time-optimal flip ==================================
python optimal_flip_analysis.py --model models/ppo_flip.zip || goto :fail

echo.
echo === 6. failure analysis =================================================
python failure_analysis.py --model models/ppo_flip.zip || goto :fail

echo.
echo ALL DONE - see results\depth\
goto :eof

:fail
echo.
echo *** A step failed. Fix the error above and run run_all.bat again: finished trainings are skipped. ***
exit /b 1
