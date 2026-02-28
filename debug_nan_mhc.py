#!/usr/bin/env python3
"""Debug script for NaN issues in mHC training."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from src.models import ResTranOCR_mHC


def test_nan_stability():
    """Test if mHC model is numerically stable."""
    print("=" * 80)
    print("Testing mHC Numerical Stability")
    print("=" * 80)
    print()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Create model
    model = ResTranOCR_mHC(
        num_classes=37,
        transformer_heads=8,
        transformer_layers=3,
        use_stn=True,
        mhc_n=4
    ).to(device)
    
    # Create dummy data
    batch_size = 4
    num_frames = 5
    x = torch.randn(batch_size, num_frames, 3, 32, 128).to(device)
    
    print("1. Testing forward pass...")
    try:
        model.eval()
        with torch.no_grad():
            output = model(x)
        
        has_nan = torch.isnan(output).any().item()
        has_inf = torch.isinf(output).any().item()
        
        if has_nan:
            print("   ❌ NaN detected in output!")
            return False
        elif has_inf:
            print("   ❌ Inf detected in output!")
            return False
        else:
            print("   ✅ Forward pass OK - No NaN/Inf")
    except Exception as e:
        print(f"   ❌ Forward pass failed: {e}")
        return False
    
    print("\n2. Testing backward pass...")
    try:
        model.train()
        output = model(x)
        loss = output.mean()
        loss.backward()
        
        # Check for NaN in gradients
        nan_grads = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    nan_grads.append(name)
        
        if nan_grads:
            print(f"   ❌ NaN detected in gradients of: {nan_grads[:5]}")
            return False
        else:
            print("   ✅ Backward pass OK - No NaN in gradients")
    except Exception as e:
        print(f"   ❌ Backward pass failed: {e}")
        return False
    
    print("\n3. Testing with extreme values...")
    try:
        # Test with very large values
        x_extreme = torch.randn(2, 5, 3, 32, 128).to(device) * 100
        with torch.no_grad():
            output = model(x_extreme)
        
        if torch.isnan(output).any() or torch.isinf(output).any():
            print("   ⚠️  Model unstable with large inputs (but protected by clipping)")
        else:
            print("   ✅ Handles extreme values OK")
    except Exception as e:
        print(f"   ⚠️  Extreme value test failed: {e}")
    
    print("\n4. Testing alpha parameters...")
    for name, param in model.named_parameters():
        if 'alpha' in name:
            print(f"   {name}: {param.item():.6f}")
    
    return True


def check_training_recommendations():
    """Print recommendations for stable training."""
    print("\n" + "=" * 80)
    print("🔧 Recommendations for Stable Training")
    print("=" * 80)
    print()
    
    print("1. Use smaller learning rate:")
    print("   python train.py --model restran_mhc --lr 1e-4  # Half of default")
    print()
    
    print("2. Increase gradient clipping:")
    print("   # In config.py:")
    print("   GRAD_CLIP: float = 1.0  # More aggressive (default: 5.0)")
    print()
    
    print("3. Use smaller batch size:")
    print("   python train.py --model restran_mhc --batch-size 32")
    print()
    
    print("4. Start with smaller n:")
    print("   python train.py --model restran_mhc --mhc-n 2  # Easier to train")
    print()
    
    print("5. Use baseline model first to verify data:")
    print("   python train.py --model restran  # No mHC")
    print()
    
    print("6. Check for bad augmentations:")
    print("   # In config.py:")
    print("   AUGMENTATION_LEVEL: str = 'light'  # Less aggressive")
    print()
    
    print("7. Monitor alpha parameters during training:")
    print("   Look for alpha values growing too large (> 0.5)")
    print()


def quick_fix_config():
    """Generate safe config for training."""
    print("\n" + "=" * 80)
    print("⚡ Quick Fix Config")
    print("=" * 80)
    print()
    
    safe_config = """
# Safe training config for mHC (add to config.py or use CLI args):

python train.py \\
    --model restran_mhc \\
    --mhc-n 2 \\
    --lr 1e-4 \\
    --batch-size 32 \\
    --epochs 50

# Or in config.py:
MODEL_TYPE: str = "restran_mhc"
MHC_N: int = 2                    # Start small
LEARNING_RATE: float = 1e-4       # Half of default
BATCH_SIZE: int = 32              # Smaller batches
GRAD_CLIP: float = 1.0            # Aggressive clipping
AUGMENTATION_LEVEL: str = "light" # Less aggressive augmentation
OVERSAMPLE_RARE_CHARS: bool = False  # Disable to simplify
"""
    print(safe_config)


def main():
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "mHC NaN Debugger" + " " * 37 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run stability tests
    stable = test_nan_stability()
    
    # Print recommendations
    check_training_recommendations()
    quick_fix_config()
    
    print("=" * 80)
    if stable:
        print("✅ Model is numerically stable!")
        print("   The NaN issue during training is likely from:")
        print("   1. Learning rate too high")
        print("   2. Gradient explosion (need more aggressive clipping)")
        print("   3. Bad data augmentation creating extreme values")
    else:
        print("❌ Model has numerical instability!")
        print("   Please use the safe config above and monitor for warnings.")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
