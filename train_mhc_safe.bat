@echo off
REM Safe training config for mHC to prevent NaN
REM Based on debug results showing model is numerically stable

echo ============================================================
echo   Training ResTranOCR_mHC with Safe Configuration
echo ============================================================
echo.
echo Changes from default:
echo   - Learning rate: 5e-4 -^> 1e-4 (50%% reduction)
echo   - Gradient clip: 5.0 -^> 1.0 (80%% more aggressive)
echo   - Batch size: 64 -^> 32 (50%% reduction for stability)
echo   - mHC streams: 4 -^> 2 (simpler, easier to train)
echo   - Augmentation: full -^> light (less extreme values)
echo   - Oversampling: OFF (simplified dataset)
echo.
echo Starting training...
echo.

python train.py ^
    --model restran_mhc ^
    --mhc-n 2 ^
    --lr 1e-4 ^
    --batch-size 32 ^
    --epochs 50 ^
    --aug-level light ^
    -n restran_mhc_safe

echo.
echo ============================================================
echo   Training completed!
echo ============================================================
echo.
pause
