@echo off
setlocal
cd /d "%~dp0"

echo =====================================================
echo CV7001 - Image-Based Damage F1 Evaluation
echo =====================================================
echo Project folder: %CD%
echo.

if not exist "CV7001_evaluate_f1.py" goto missing_script
if not exist "CV7001_F1_study_config_PRIVATE.json" goto missing_config
if not exist "CV7001_F1_input\" goto missing_images

py -3 --version >nul 2>&1
if not errorlevel 1 (
    echo Running with Python Launcher...
    py -3 "CV7001_evaluate_f1.py" --config "CV7001_F1_study_config_PRIVATE.json" --input-dir "CV7001_F1_input" --output-dir "CV7001_F1_results" --label-mode damage --no-sensitivity
) else (
    echo Python Launcher unavailable. Trying python...
    python "CV7001_evaluate_f1.py" --config "CV7001_F1_study_config_PRIVATE.json" --input-dir "CV7001_F1_input" --output-dir "CV7001_F1_results" --label-mode damage --no-sensitivity
)

if errorlevel 1 goto run_failed

echo.
echo SUCCESS: Evaluation finished.
echo See: CV7001_F1_results\f1_comparison.csv
echo Review *_damage_review.png and *_damage_alignment_check.png.
echo Both use the same F1 ground-truth mask; *_GT_mask_rectified.png is the raw mask.
echo IMPORTANT: New image versions may produce different F1 scores.
echo.
pause
exit /b 0

:missing_script
echo ERROR: CV7001_evaluate_f1.py not found.
goto failure
:missing_config
echo ERROR: CV7001_F1_study_config_PRIVATE.json not found.
goto failure
:missing_images
echo ERROR: CV7001_F1_input folder not found.
goto failure
:run_failed
echo ERROR: Python evaluation failed. Review the error above.
echo Install Python libraries with: py -3 -m pip install -r requirements.txt
:failure
echo.
pause
exit /b 1
