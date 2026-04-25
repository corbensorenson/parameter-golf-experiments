@echo off
set "ROOT=C:\Users\corbe\Documents\golf\workspace\parameter-golf"
set "PY=C:\Users\corbe\Documents\golf\workspace\parameter-golf\.venv-cuda313\Scripts\python.exe"
set "SCRIPT=C:\Users\corbe\Documents\golf\workspace\parameter-golf\scripts\run_caseops_candidate_2060_compare.py"
set "LOGROOT=C:\Users\corbe\Documents\golf\logs"
set "RUN_ID=caseops-candidate2060-compare20k-20260423-063700"
set "STDOUT=C:\Users\corbe\Documents\golf\logs\caseops-candidate2060-compare20k-20260423-063700.txt"
set "STDERR=C:\Users\corbe\Documents\golf\logs\caseops-candidate2060-compare20k-20260423-063700.err.txt"
set "STATUS=C:\Users\corbe\Documents\golf\logs\caseops-candidate2060-compare20k-live.status.txt"
set "CANDIDATE2060_PROFILE=compare20k"
set "PYTHONUNBUFFERED=1"
(
  echo run_id=%RUN_ID%
  echo stdout=%STDOUT%
  echo stderr=%STDERR%
  echo started=2026-04-23T06:37:00-05:00
  echo launcher=cmd_start_detached
) > "%STATUS%"
cd /d "%ROOT%"
start "caseops20k" /min cmd.exe /c ""%PY%" -u "%SCRIPT%" 1> "%STDOUT%" 2> "%STDERR%""
