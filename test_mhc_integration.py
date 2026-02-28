#!/usr/bin/env python3
"""Quick test to verify mHC model integration."""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from src.models import ResTranOCR, ResTranOCR_mHC

def test_model_initialization():
    """Test if models can be initialized properly."""
    print("=" * 80)
    print("Testing Model Initialization")
    print("=" * 80)
    
    num_classes = 37
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\nUsing device: {device}")
    print()
    
    # Test baseline
    print("1. Testing ResTranOCR (baseline)...")
    try:
        model_baseline = ResTranOCR(
            num_classes=num_classes,
            transformer_heads=8,
            transformer_layers=3,
            use_stn=True
        ).to(device)
        params_baseline = sum(p.numel() for p in model_baseline.parameters())
        print(f"   ✓ Initialized: {params_baseline:,} parameters")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test mHC variants
    for n in [2, 4, 8]:
        print(f"\n2. Testing ResTranOCR_mHC (n={n})...")
        try:
            model_mhc = ResTranOCR_mHC(
                num_classes=num_classes,
                transformer_heads=8,
                transformer_layers=3,
                use_stn=True,
                mhc_n=n
            ).to(device)
            params_mhc = sum(p.numel() for p in model_mhc.parameters())
            increase = params_mhc - params_baseline
            increase_pct = (increase / params_baseline) * 100
            print(f"   ✓ Initialized: {params_mhc:,} parameters (+{increase_pct:.1f}%)")
        except Exception as e:
            print(f"   ✗ Failed: {e}")
            return False
    
    return True


def test_forward_pass():
    """Test if models can perform forward pass."""
    print("\n" + "=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    
    batch_size = 2
    num_frames = 5
    height = 32
    width = 128
    num_classes = 37
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy input
    x = torch.randn(batch_size, num_frames, 3, height, width).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Test baseline
    print("\n1. Testing baseline forward pass...")
    try:
        model = ResTranOCR(num_classes=num_classes, use_stn=True).to(device)
        with torch.no_grad():
            output = model(x)
        print(f"   ✓ Output shape: {output.shape}")
        print(f"   Expected: [batch={batch_size}, seq_len, classes={num_classes}]")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        return False
    
    # Test mHC
    print("\n2. Testing mHC forward pass (n=4)...")
    try:
        model_mhc = ResTranOCR_mHC(num_classes=num_classes, use_stn=True, mhc_n=4).to(device)
        with torch.no_grad():
            output_mhc = model_mhc(x)
        print(f"   ✓ Output shape: {output_mhc.shape}")
        print(f"   Shape match with baseline: {output.shape == output_mhc.shape}")
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def test_backward_pass():
    """Test if models can perform backward pass."""
    print("\n" + "=" * 80)
    print("Testing Backward Pass")
    print("=" * 80)
    
    batch_size = 2
    num_frames = 5
    height = 32
    width = 128
    num_classes = 37
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\nTesting gradient flow through mHC model...")
    try:
        model = ResTranOCR_mHC(num_classes=num_classes, use_stn=True, mhc_n=4).to(device)
        x = torch.randn(batch_size, num_frames, 3, height, width).to(device)
        
        # Forward
        output = model(x)
        
        # Dummy loss
        loss = output.mean()
        
        # Backward
        loss.backward()
        
        # Check if gradients exist
        has_grads = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        
        if has_grads:
            print("   ✓ Gradients computed successfully")
            
            # Check mHC specific parameters have gradients
            mhc_params_with_grad = 0
            for name, param in model.named_parameters():
                if 'mhc' in name and param.requires_grad and param.grad is not None:
                    mhc_params_with_grad += 1
            
            print(f"   ✓ mHC parameters with gradients: {mhc_params_with_grad}")
        else:
            print("   ✗ No gradients computed")
            return False
            
    except Exception as e:
        print(f"   ✗ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 25 + "mHC Integration Test" + " " * 33 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    tests = [
        ("Model Initialization", test_model_initialization),
        ("Forward Pass", test_forward_pass),
        ("Backward Pass", test_backward_pass),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"\n❌ Test '{test_name}' crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status}  {test_name}")
    
    all_passed = all(success for _, success in results)
    
    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 All tests passed! mHC integration is working correctly.")
        print("\nYou can now:")
        print("  1. Train with: python train.py --model restran_mhc --mhc-n 4")
        print("  2. Run ablation: python run_mhc_ablation.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    print("=" * 80)
    print()
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
