#!/usr/bin/env python3
"""Quick training script for mHC ablation study."""
import os
import subprocess
import sys

def run_experiment(model_type, mhc_n=None, epochs=30, experiment_suffix=""):
    """Run single training experiment."""
    cmd = [
        sys.executable, "train.py",
        "--model", model_type,
        "--epochs", str(epochs),
        "--batch-size", "32",  # Smaller for faster iteration
        "--lr", "5e-4",
    ]
    
    # Add mHC specific args
    if mhc_n is not None:
        cmd.extend(["--mhc-n", str(mhc_n)])
        experiment_name = f"{model_type}_n{mhc_n}{experiment_suffix}"
    else:
        experiment_name = f"{model_type}{experiment_suffix}"
    
    cmd.extend(["-n", experiment_name])
    
    print("=" * 80)
    print(f"🚀 Running experiment: {experiment_name}")
    print("=" * 80)
    print(f"Command: {' '.join(cmd)}")
    print()
    
    subprocess.run(cmd, check=True)
    print()


def main():
    """Run mHC ablation study."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "mHC Ablation Study - Training Runner" + " " * 22 + "║")
    print("╚" + "=" * 78 + "╝")
    print()
    
    # Configuration
    EPOCHS = 30  # Quick test (use 80 for full training)
    
    experiments = [
        # Baseline
        ("restran", None, "Baseline (no mHC)"),
        
        # mHC with different stream counts
        ("restran_mhc", 2, "mHC with 2 streams"),
        ("restran_mhc", 4, "mHC with 4 streams (recommended)"),
        ("restran_mhc", 8, "mHC with 8 streams"),
    ]
    
    print("📋 Experiments to run:")
    for i, (model, mhc_n, desc) in enumerate(experiments, 1):
        print(f"   {i}. {desc}")
    print()
    
    response = input("Run all experiments? [y/N]: ").strip().lower()
    if response != 'y':
        print("Cancelled.")
        return
    
    print("\n" + "=" * 80)
    print("Starting ablation study...")
    print("=" * 80 + "\n")
    
    for model_type, mhc_n, desc in experiments:
        try:
            run_experiment(
                model_type=model_type,
                mhc_n=mhc_n,
                epochs=EPOCHS,
                experiment_suffix="_ablation"
            )
        except subprocess.CalledProcessError as e:
            print(f"❌ Experiment failed: {desc}")
            print(f"Error: {e}")
            response = input("Continue with next experiment? [y/N]: ").strip().lower()
            if response != 'y':
                print("Stopping ablation study.")
                return
    
    print("\n" + "=" * 80)
    print("✅ Ablation study completed!")
    print("=" * 80)
    print()
    print("Check results in:")
    print("  - results/ directory for checkpoints")
    print("  - Trainer logs for metrics")
    print()
    print("To compare results, run:")
    print("  python -m src.utils.visualize")
    print()


if __name__ == "__main__":
    main()
