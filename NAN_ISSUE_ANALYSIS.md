# NaN Training Issue - Root Cause Analysis & Solution

## 🔴 Problem Summary

Your training at epoch 29 shows `loss=nan` continuously with `lr=4.92e-04`. This indicates:

1. **The checkpoint is corrupted** - Once NaN enters model parameters, it propagates through every batch forever
2. **Wrong learning rate** - The config was set to 1e-4, but training shows 4.92e-04, meaning:
   - Training was started with old config (LR=5e-4)
   - OneCycleLR scheduler state was saved in checkpoint
   - Changing config mid-training doesn't affect loaded scheduler

## 🔍 Root Cause Traced

### Issue Chain:
```
Epoch ~28: Gradient explosion with LR=5e-4 + aggressive settings
    ↓
NaN enters one or more parameters (likely mHC alpha or routing matrices)
    ↓
Checkpoint saved with NaN parameters
    ↓
Every subsequent batch: NaN input → NaN output → NaN gradients → NaN parameters
    ↓
Training becomes unrecoverable
```

### Why mHC is More Sensitive:
- **Extra learnable parameters**: Alpha gates (α_pre, α_post, α_res) multiply routing matrices
- **Sinkhorn-Knopp algorithm**: Iterative normalization can explode if inputs are large
- **Einsum operations**: `torch.einsum('bnn,bnchw->bnchw')` amplifies small errors
- **Multi-stream aggregation**: Errors from n=4 streams compound

## ✅ Solutions Implemented

### 1. **Comprehensive NaN Detection** (Added to code)

#### In `src/models/components.py`:
- **5 checkpoints** in mHC forward pass:
  - ① After pooling/normalization
  - ② After read-out aggregation
  - ③ After sublayer (CNN block) output
  - ④ After write-in + residual
  - ⑤ Final output validation

- **Detailed logging** when NaN detected:
  ```
  🔴 NaN/Inf in mHC INPUT (pooled):
     Shape: [batch, n*channels]
     NaN count: 127
     Min/Max values of non-NaN elements
     Alpha parameters: pre=X, post=Y, res=Z
  ```

#### In `src/training/trainer.py`:
- **Pre-epoch parameter check**: Verify model health before starting epoch
- **Input data validation**: Catch corrupt batches
- **Prediction validation**: Detect NaN immediately after forward pass
- **Gradient checking**: Inspect all parameter gradients before optimization
- **Loss validation**: Stop training immediately if loss becomes NaN

### 2. **Diagnostic Tool** - `diagnose_nan.py`

Run to check model health:
```bash
python diagnose_nan.py
```

What it does:
- ✅ Checks if config is correct
- ✅ Scans for existing checkpoints and tests if corrupted
- ✅ Tests fresh model initialization
- ✅ Verifies forward pass works
- ✅ Displays alpha parameter values
- ✅ Provides recommendations

### 3. **Updated Config** (Already done in previous step)

Safe hyperparameters for mHC training:
```python
MODEL_TYPE = "restran_mhc"
LEARNING_RATE = 1e-4          # 50% lower than baseline
BATCH_SIZE = 32               # 50% smaller
GRAD_CLIP = 1.0               # 5x more aggressive
MHC_N = 2                     # Fewer streams for stability
AUGMENTATION_LEVEL = "light"  # Less data perturbation
OVERSAMPLE_RARE_CHARS = False # Reduce training variance
```

## 📋 Next Steps

### **MUST DO FIRST: Stop Current Training & Clean Up**

```powershell
# 1. Stop the current training (Ctrl+C if still running)

# 2. Backup corrupted checkpoints
New-Item -ItemType Directory -Force results/corrupted_backup
Move-Item results/*.pth results/corrupted_backup/

# 3. Run diagnostic to verify clean state
python diagnose_nan.py
```

### **Start Fresh Training with Debugging**

```bash
# Activate environment
conda activate aic25

# Start training with safe hyperparameters
python train.py --model restran_mhc --mhc-n 2 --lr 0.0001

# Or use config defaults (already set correctly)
python train.py
```

### **What Will Happen with New Training:**

The enhanced logging will now catch NaN **at the exact moment it occurs**:

**Example output when NaN detected:**
```
🔍 Checking model parameters before epoch 15...
✅ Model parameters OK

Ep 15/80:  45%|████▌     | 268/594 [01:52<02:17,  2.37it/s, loss=2.341, lr=1.23e-04]

🔴 NaN/Inf detected in MODEL PREDICTIONS:
   Pred shape: torch.Size([32, 26, 37])
   NaN count: 2464
   Inf count: 0
   This indicates NaN originated in forward pass

🟡 NaN/Inf in SUBLAYER OUTPUT (h_out):
   Sublayer: BasicBlock(...)
   h_out: NaN=1024, Inf=0
   h_out stats (non-nan): min=-3.456, max=5.234

   Stopping training to prevent corruption...
```

This tells you:
- **Which epoch/batch** NaN first appeared
- **Which layer/component** caused it
- **What the values were** before NaN

## 🧪 Testing Recommendations

### Stage 1: Validate Stability (Epochs 1-5)
Monitor for:
- Loss decreasing normally (should start ~10-15, drop to ~5)
- No NaN warnings printed
- Alpha parameters staying small (<0.1)

### Stage 2: Continued Training (Epochs 6-30)
If Stage 1 passes:
- Training should be stable
- Loss continues decreasing
- Validation accuracy starts improving

### Stage 3: Scale Up (After successful run)
Once stable training confirmed, you can try:
- Increase `MHC_N` from 2 → 4
- Increase `LEARNING_RATE` from 1e-4 → 1.5e-4
- Increase `BATCH_SIZE` from 32 → 48

## 📊 Expected Behavior vs Issues

### ✅ Healthy Training:
```
Ep 1/80: 100%|████████| 594/594 [04:12<00:00, 2.35it/s, loss=12.456, lr=5.00e-05]
🔍 Checking model parameters before epoch 2...
✅ Model parameters OK

Ep 2/80: 100%|████████| 594/594 [04:11<00:00, 2.36it/s, loss=8.234, lr=7.23e-05]
```

### 🔴 NaN Detected (with new logging):
```
Ep 15/80:  45%|████▌     | 268/594 [01:52<02:17,  2.37it/s, loss=2.341]

🟡 NaN/Inf in mHC READ-OUT (h_in):
   H_pre stats: min=0.123456, max=0.876543, mean=0.498765
   h_in: NaN=512, Inf=0

ValueError: NaN detected in model predictions - forward pass corrupted
```

## 🎯 Success Criteria

You'll know the fix worked when:
1. ✅ `diagnose_nan.py` reports model healthy
2. ✅ Training starts from epoch 1 with no checkpoints
3. ✅ Loss decreases smoothly for first 5 epochs
4. ✅ No NaN warnings printed during training
5. ✅ Validation accuracy improves over time

## ⚠️ If NaN Still Occurs

Even with safe hyperparameters, if you see NaN:

### Immediate Actions:
1. **Check the NaN log output** - It will tell you exactly which layer/component
2. **Report the specific error** - The detailed stats help identify the issue
3. **Try even more conservative settings**:
   ```python
   MHC_N = 1  # Essentially disables multi-stream
   LEARNING_RATE = 5e-5  # Half again
   BATCH_SIZE = 16  # Half again
   ```

### Possible Additional Causes:
- **Data corruption**: Some images may contain NaN/Inf → Input validation will catch this
- **Hardware issues**: GPU memory corruption → Try `USE_CUDNN_BENCHMARK = False`
- **Numerical precision**: Mixed precision issues → The code already uses `GradScaler`

## 📝 Files Modified

All changes for NaN detection:

1. **src/models/components.py** (lines ~450-550):
   - Enhanced `apply_mhc()` with 5 checkpoints
   - Detailed statistics logging
   - Safe-exit on NaN

2. **src/training/trainer.py** (lines ~130-210):
   - Pre-epoch parameter validation
   - Input/prediction/gradient checking
   - Early stopping on NaN

3. **diagnose_nan.py** (new file):
   - Checkpoint corruption detection
   - Model initialization testing
   - Alpha parameter monitoring

4. **configs/config.py** (lines 20-85):
   - Safe hyperparameters for mHC

## 🔗 Architecture Details

For understanding why mHC needs careful tuning:

```
Input [B,C,H,W]
    ↓ Expand to n streams
[B,n,C,H,W]
    ↓ Global pool → [B,n*C]
    ↓ Generate routing coefficients via phi(RMSNorm(...))
    ↓ Project to manifolds (sigmoid, sinkhorn_knopp)
    ↓ H_pre: Read-out [B,n] → Aggregate to [B,C,H,W]
    ↓ Apply sublayer (ResNet block)
    ↓ H_post: Write-in [B,n] → Distribute to [B,n,C,H,W]
    ↓ H_res: Stream mixing [B,n,n] → Update streams
    ↓ Collapse streams (mean)
Output [B,C,H,W]
```

Critical numerical operations:
- `RMSNorm(eps=1e-20)`: Very small eps can cause division issues
- `exp(H_tilde)` in Sinkhorn: Can overflow if H_tilde > 10
- `einsum('bnn,bnchw->bnchw')`: Amplifies errors
- `alpha * H_tilde`: If alpha grows > 1.0, routing becomes unstable

All now protected with clipping and NaN checks!

## ✨ Summary

**The old training cannot be recovered** - the checkpoint has NaN.

**The solution is a fresh start** with:
1. ✅ Safe hyperparameters (already configured)
2. ✅ Comprehensive NaN detection (already implemented)
3. ✅ Diagnostic tools (already created)

**Next action:** Run `python diagnose_nan.py`, then start fresh training.

The enhanced debugging will now **catch the exact moment and location** where NaN originates, allowing for precise fixes if it happens again.
