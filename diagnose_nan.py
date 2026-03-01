#!/usr/bin/env python3
"""
Diagnostic script to trace NaN issues in mHC training.

This script checks for pre-existing NaN in checkpoints and provides
recommendations for resolving the issue.
"""
import os
import sys
import torch
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from configs.config import Config
from src.models.restran import ResTranOCR_mHC


def check_checkpoint(checkpoint_path: str):
    """Check if checkpoint contains NaN parameters."""
    print(f"\n🔍 Analyzing checkpoint: {checkpoint_path}")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Check state dict
    state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    has_nan = False
    nan_params = []
    
    for name, param in state_dict.items():
        if torch.is_tensor(param):
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_count = torch.isnan(param).sum().item()
                inf_count = torch.isinf(param).sum().item()
                nan_params.append((name, param.shape, nan_count, inf_count))
                has_nan = True
    
    if has_nan:
        print(f"\n🔴 CHECKPOINT IS CORRUPTED - Contains NaN/Inf:")
        print(f"   Found {len(nan_params)} corrupted parameters:")
        for name, shape, nan_count, inf_count in nan_params[:10]:  # Show first 10
            print(f"   - {name}: shape={shape}, NaN={nan_count}, Inf={inf_count}")
        if len(nan_params) > 10:
            print(f"   ... and {len(nan_params) - 10} more")
        return False
    else:
        print(f"✅ Checkpoint is HEALTHY - No NaN/Inf detected")
        
        # Show additional info
        if 'epoch' in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if 'best_acc' in checkpoint:
            print(f"   Best Accuracy: {checkpoint['best_acc']:.2f}%")
        if 'optimizer_state_dict' in checkpoint:
            print(f"   Contains optimizer state: Yes")
        if 'scheduler_state_dict' in checkpoint:
            print(f"   Contains scheduler state: Yes")
        
        return True


def check_model_initialization():
    """Check if fresh model initialization is healthy."""
    print(f"\n🔍 Testing fresh model initialization...")
    
    config = Config()
    
    try:
        model = ResTranOCR_mHC(
            num_classes=config.NUM_CLASSES,
            mhc_n=config.MHC_N
        )
        
        # Check all parameters
        nan_params = []
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                nan_params.append(name)
        
        if nan_params:
            print(f"❌ Fresh model has NaN/Inf in {len(nan_params)} parameters:")
            for name in nan_params[:5]:
                print(f"   - {name}")
            return False
        
        # Test forward pass
        print(f"   Testing forward pass...")
        # ResTranOCR_mHC expects [B, F, C, H, W] - Batch, Frames, Channels, Height, Width
        dummy_input = torch.randn(2, 5, 3, 32, 128)  # [B, num_frames, C, H, W]
        
        with torch.no_grad():
            output = model(dummy_input)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print(f"❌ Forward pass produces NaN/Inf")
            return False
        
        print(f"✅ Fresh model initialization is HEALTHY")
        print(f"   Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   mHC streams (n): {config.MHC_N}")
        
        # Check alpha parameters
        print(f"\n   mHC alpha parameters:")
        if hasattr(model.backbone, 'mhc_1'):
            for mhc_idx in range(1, 5):  # mhc_1, mhc_2, mhc_3, mhc_4
                mhc_attr = f'mhc_{mhc_idx}'
                if hasattr(model.backbone, mhc_attr):
                    mhc_wrapper = getattr(model.backbone, mhc_attr)
                    alpha_pre = mhc_wrapper.alpha_pre.item()
                    alpha_post = mhc_wrapper.alpha_post.item()
                    alpha_res = mhc_wrapper.alpha_res.item()
                    layer_name = f'layer{mhc_idx}'
                    print(f"   {layer_name}: pre={alpha_pre:.6f}, post={alpha_post:.6f}, res={alpha_res:.6f}")
        else:
            print(f"   (No mHC wrappers found - model may not be using mHC)")
        
        return True
    
    except Exception as e:
        print(f"❌ Error during model initialization: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main diagnostic routine."""
    print("="*80)
    print("NaN DIAGNOSTIC TOOL FOR mHC TRAINING")
    print("="*80)
    
    config = Config()
    
    # 1. Check config
    print(f"\n📋 Current Configuration:")
    print(f"   MODEL_TYPE: {config.MODEL_TYPE}")
    print(f"   MHC_N: {config.MHC_N}")
    print(f"   LEARNING_RATE: {config.LEARNING_RATE}")
    print(f"   BATCH_SIZE: {config.BATCH_SIZE}")
    print(f"   GRAD_CLIP: {config.GRAD_CLIP}")
    print(f"   AUGMENTATION_LEVEL: {config.AUGMENTATION_LEVEL}")
    print(f"   OVERSAMPLE_RARE_CHARS: {config.OVERSAMPLE_RARE_CHARS}")
    
    # 2. Check for existing checkpoints
    print(f"\n\n🔍 Checking for existing checkpoints...")
    results_dir = Path("results")
    
    if results_dir.exists():
        checkpoints = list(results_dir.glob("*.pth"))
        if checkpoints:
            print(f"   Found {len(checkpoints)} checkpoint(s):")
            
            healthy_checkpoints = []
            corrupted_checkpoints = []
            
            for ckpt in sorted(checkpoints):
                is_healthy = check_checkpoint(str(ckpt))
                if is_healthy:
                    healthy_checkpoints.append(ckpt)
                else:
                    corrupted_checkpoints.append(ckpt)
            
            print(f"\n📊 Checkpoint Summary:")
            print(f"   Healthy: {len(healthy_checkpoints)}")
            print(f"   Corrupted: {len(corrupted_checkpoints)}")
            
            if corrupted_checkpoints:
                print(f"\n⚠️  WARNING: Training from corrupted checkpoints will fail immediately!")
                print(f"   Corrupted checkpoints should be backed up and removed.")
        else:
            print(f"   No checkpoints found in {results_dir}")
    else:
        print(f"   Results directory doesn't exist yet: {results_dir}")
    
    # 3. Test fresh model initialization
    print(f"\n" + "="*80)
    model_healthy = check_model_initialization()
    
    # 4. Provide recommendations
    print(f"\n" + "="*80)
    print(f"RECOMMENDATIONS:")
    print(f"="*80)
    
    if 'corrupted_checkpoints' in locals() and corrupted_checkpoints:
        print(f"\n🔴 CRITICAL: Your checkpoint is corrupted with NaN")
        print(f"\n   To fix:")
        print(f"   1. STOP the current training (Ctrl+C)")
        print(f"   2. Backup corrupted checkpoints:")
        print(f"      mkdir results/corrupted_backup")
        print(f"      mv results/*.pth results/corrupted_backup/")
        print(f"   3. Start fresh training:")
        print(f"      python train.py --model restran_mhc --mhc-n 2")
    
    elif model_healthy:
        print(f"\n✅ Model architecture is healthy")
        print(f"\n   Safe to start training:")
        print(f"   python train.py --model restran_mhc --mhc-n {config.MHC_N} --lr {config.LEARNING_RATE}")
        print(f"\n   With the new debugging, training will:")
        print(f"   - Check model parameters at start of each epoch")
        print(f"   - Verify input data doesn't contain NaN")
        print(f"   - Check predictions after forward pass")
        print(f"   - Examine gradients before optimization")
        print(f"   - Stop immediately if NaN detected with detailed diagnostics")
    
    else:
        print(f"\n🔴 Model initialization has issues")
        print(f"   This suggests a code problem. Check:")
        print(f"   - src/models/components.py (mHC implementation)")
        print(f"   - src/models/restran.py (ResTranOCR_mHC)")
    
    print(f"\n" + "="*80)
    print(f"For questions, check the detailed logging added to:")
    print(f"  - src/models/components.py (mHC NaN detection)")
    print(f"  - src/training/trainer.py (gradient/parameter checking)")
    print(f"="*80 + "\n")


if __name__ == "__main__":
    main()
