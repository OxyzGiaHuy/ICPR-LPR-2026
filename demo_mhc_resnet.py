"""
Demo script for ResNet with Manifold-Constrained Hyper-Connections (mHC).

This script demonstrates:
1. How to instantiate ResTranOCR_mHC model
2. Compare parameters and FLOPs with standard ResTranOCR
3. Test forward pass with dummy data
4. Visualize the mHC mechanism benefits

Based on DeepSeek-V3 architecture, adapted for CNN residual connections.
"""
import torch
import torch.nn as nn
from src.models import ResTranOCR, ResTranOCR_mHC


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_forward_pass():
    """Test forward pass with dummy data."""
    print("=" * 80)
    print("Testing Forward Pass")
    print("=" * 80)
    
    # Configuration
    num_classes = 37  # 26 letters + 10 digits + blank
    batch_size = 2
    num_frames = 5
    height = 64
    width = 128
    
    # Create dummy input
    x = torch.randn(batch_size, num_frames, 3, height, width)
    print(f"Input shape: {x.shape}")
    print(f"  Batch size: {batch_size}")
    print(f"  Frames: {num_frames}")
    print(f"  Image size: {height}x{width}")
    print()
    
    # Standard ResTranOCR
    print("-" * 80)
    print("1. Standard ResTranOCR (Baseline)")
    print("-" * 80)
    model_standard = ResTranOCR(
        num_classes=num_classes,
        transformer_heads=8,
        transformer_layers=3,
        transformer_ff_dim=2048,
        dropout=0.1,
        use_stn=True
    )
    
    with torch.no_grad():
        output_standard = model_standard(x)
    
    params_standard = count_parameters(model_standard)
    print(f"Parameters: {params_standard:,}")
    print(f"Output shape: {output_standard.shape}")
    print(f"  Sequence length: {output_standard.shape[1]}")
    print(f"  Num classes: {output_standard.shape[2]}")
    print()
    
    # ResTranOCR with mHC
    print("-" * 80)
    print("2. ResTranOCR with mHC (n=4 streams)")
    print("-" * 80)
    model_mhc = ResTranOCR_mHC(
        num_classes=num_classes,
        transformer_heads=8,
        transformer_layers=3,
        transformer_ff_dim=2048,
        dropout=0.1,
        use_stn=True,
        mhc_n=4  # 4 parallel streams
    )
    
    with torch.no_grad():
        output_mhc = model_mhc(x)
    
    params_mhc = count_parameters(model_mhc)
    print(f"Parameters: {params_mhc:,}")
    print(f"Output shape: {output_mhc.shape}")
    print(f"  Sequence length: {output_mhc.shape[1]}")
    print(f"  Num classes: {output_mhc.shape[2]}")
    print()
    
    # Compare models
    print("-" * 80)
    print("3. Comparison")
    print("-" * 80)
    param_increase = params_mhc - params_standard
    param_increase_pct = (param_increase / params_standard) * 100
    
    print(f"Parameter increase: {param_increase:,} (+{param_increase_pct:.2f}%)")
    print(f"Output shape match: {output_standard.shape == output_mhc.shape}")
    print()
    
    # Test different mHC configurations
    print("-" * 80)
    print("4. Testing different mHC stream configurations")
    print("-" * 80)
    for n in [2, 4, 8]:
        model = ResTranOCR_mHC(
            num_classes=num_classes,
            transformer_heads=8,
            transformer_layers=3,
            use_stn=True,
            mhc_n=n
        )
        params = count_parameters(model)
        param_increase = params - params_standard
        param_increase_pct = (param_increase / params_standard) * 100
        print(f"  n={n} streams: {params:,} parameters (+{param_increase_pct:.2f}%)")
    print()


def explain_mhc_mechanism():
    """Explain mHC mechanism with visualization."""
    print("=" * 80)
    print("Understanding mHC Mechanism")
    print("=" * 80)
    print()
    print("Standard ResNet Residual Connection:")
    print("  output = conv_layers(input) + input")
    print()
    print("mHC-Enhanced Residual Connection:")
    print("  1. Expand: input -> [stream_1, stream_2, ..., stream_n]")
    print("  2. Process: each stream goes through conv_layers")
    print("  3. Combine: streams combined using doubly stochastic matrices")
    print("     - H_pre: read-out weights (aggregate streams for CNN input)")
    print("     - H_post: write-in weights (distribute CNN output to streams)")
    print("     - H_res: residual routing matrix (stream-to-stream connections)")
    print()
    print("Benefits:")
    print("  ✓ Multi-stream processing enables diverse feature learning")
    print("  ✓ Doubly stochastic matrices ensure stable gradient flow")
    print("  ✓ Adaptive gating learns optimal feature combination")
    print("  ✓ Different streams can specialize for different conditions:")
    print("    - Stream 1: Good lighting (Scenario A)")
    print("    - Stream 2: Low light / Night (Scenario B)")
    print("    - Stream 3: Rain / Weather variations")
    print("    - Stream 4: Different viewing angles")
    print()
    print("For License Plate Recognition:")
    print("  → Better handling of diverse capture conditions")
    print("  → Improved generalization across scenarios")
    print("  → More robust to environmental variations")
    print()


def integration_example():
    """Show how to integrate mHC model in training."""
    print("=" * 80)
    print("Integration Example: Using mHC in Training")
    print("=" * 80)
    print()
    print("# In your config file (configs/config.py):")
    print("""
config = {
    'model': 'ResTranOCR_mHC',  # Use mHC-enhanced model
    'model_args': {
        'num_classes': 37,
        'transformer_heads': 8,
        'transformer_layers': 3,
        'transformer_ff_dim': 2048,
        'dropout': 0.1,
        'use_stn': True,
        'mhc_n': 4,  # Number of streams (2, 4, or 8 recommended)
    }
}
""")
    print()
    print("# In your training script:")
    print("""
from src.models import ResTranOCR_mHC

# Create model
model = ResTranOCR_mHC(**config['model_args'])

# Training loop (same as before)
for batch in dataloader:
    images, targets, target_lengths = batch
    
    # Forward pass
    logits = model(images)  # [B, T, num_classes]
    log_probs = logits.log_softmax(2)
    
    # CTC Loss
    input_lengths = torch.full((B,), T, dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lengths, target_lengths)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
""")
    print()


def ablation_study_suggestion():
    """Suggest ablation study configurations."""
    print("=" * 80)
    print("Suggested Ablation Study Configurations")
    print("=" * 80)
    print()
    print("To evaluate mHC effectiveness, run experiments with:")
    print()
    print("1. Baseline (no mHC):")
    print("   model = ResTranOCR(num_classes=37, ...)")
    print()
    print("2. mHC with 2 streams:")
    print("   model = ResTranOCR_mHC(num_classes=37, mhc_n=2, ...)")
    print()
    print("3. mHC with 4 streams (recommended):")
    print("   model = ResTranOCR_mHC(num_classes=37, mhc_n=4, ...)")
    print()
    print("4. mHC with 8 streams:")
    print("   model = ResTranOCR_mHC(num_classes=37, mhc_n=8, ...)")
    print()
    print("Expected results:")
    print("  - mHC should improve accuracy on Scenario B (challenging conditions)")
    print("  - Minimal parameter overhead (~5-10% increase)")
    print("  - Better cross-scenario generalization")
    print()
    print("Evaluation metrics:")
    print("  - Character accuracy")
    print("  - Plate accuracy")
    print("  - Scenario A vs Scenario B performance gap")
    print("  - Inference time")
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 15 + "ResNet with mHC - Demo & Guide" + " " * 32 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Run demonstrations
    explain_mhc_mechanism()
    test_forward_pass()
    integration_example()
    ablation_study_suggestion()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("  1. Run ablation study with different mhc_n values")
    print("  2. Compare performance on Scenario A vs Scenario B")
    print("  3. Analyze which streams specialize for which conditions")
    print("  4. Monitor training stability and convergence")
    print()
