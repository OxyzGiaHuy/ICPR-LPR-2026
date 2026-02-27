"""Visualization utilities for error analysis."""
import os
from typing import List, Tuple, Dict

import cv2
import numpy as np
import torch


def visualize_errors(
    images: torch.Tensor,
    predictions: List[str],
    ground_truths: List[str],
    confidences: List[float],
    track_ids: List[str],
    output_path: str,
    max_samples: int = 32,
    sort_by: str = "wrong_first"  # "wrong_first", "low_conf", or "all"
):
    """
    Visualize model predictions vs ground truth for error analysis.
    
    Args:
        images: [B, F, C, H, W] input images (5 frames per sample)
        predictions: List of predicted strings
        ground_truths: List of ground truth strings
        confidences: List of prediction confidences
        track_ids: List of track identifiers
        output_path: Where to save the visualization
        max_samples: Maximum number of samples to visualize
        sort_by: How to select samples ("wrong_first", "low_conf", "all")
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("⚠️  matplotlib not available for error visualization")
        return
    
    # Move images to CPU and denormalize
    images = images.cpu()
    B = images.size(0)
    
    # Create analysis list
    samples = []
    for i in range(B):
        is_correct = predictions[i] == ground_truths[i]
        samples.append({
            'idx': i,
            'pred': predictions[i],
            'gt': ground_truths[i],
            'conf': confidences[i],
            'track_id': track_ids[i],
            'correct': is_correct,
        })
    
    # Sort samples based on strategy
    if sort_by == "wrong_first":
        # Incorrect first, then by confidence
        samples.sort(key=lambda x: (x['correct'], -x['conf']))
    elif sort_by == "low_conf":
        # Lowest confidence first
        samples.sort(key=lambda x: x['conf'])
    # else: keep original order
    
    # Select top samples
    samples = samples[:max_samples]
    
    # Compute grid size
    n_samples = len(samples)
    n_cols = min(4, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for plot_idx, sample in enumerate(samples):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        ax = axes[row, col]
        
        idx = sample['idx']
        
        # Get middle frame (frame 2 out of 0-4)
        frame = images[idx, 2]  # [C, H, W]
        
        # Denormalize (assuming ImageNet normalization)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        frame = frame * std + mean
        frame = torch.clamp(frame, 0, 1)
        
        # Convert to numpy [H, W, C]
        img_np = frame.permute(1, 2, 0).numpy()
        
        # Display image
        ax.imshow(img_np)
        ax.axis('off')
        
        # Color-coded title
        if sample['correct']:
            color = 'green'
            status = '✓'
        else:
            color = 'red'
            status = '✗'
        
        title = f"{status} Pred: {sample['pred']}\nGT: {sample['gt']}\nConf: {sample['conf']:.3f}"
        ax.set_title(title, fontsize=9, color=color, weight='bold')
        
        # Add border
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    
    # Hide empty subplots
    for plot_idx in range(n_samples, n_rows * n_cols):
        row = plot_idx // n_cols
        col = plot_idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Save
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close()
    
    # Print summary stats
    n_correct = sum(1 for s in samples if s['correct'])
    avg_conf_correct = np.mean([s['conf'] for s in samples if s['correct']]) if n_correct > 0 else 0
    avg_conf_wrong = np.mean([s['conf'] for s in samples if not s['correct']]) if n_correct < len(samples) else 0
    
    print(f"\n📸 Error visualization saved → {output_path}")
    print(f"   Samples: {n_correct}/{len(samples)} correct ({100*n_correct/len(samples):.1f}%)")
    if n_correct > 0:
        print(f"   Avg conf (correct): {avg_conf_correct:.3f}")
    if n_correct < len(samples):
        print(f"   Avg conf (wrong):   {avg_conf_wrong:.3f}")


def analyze_confusion_pairs(
    predictions: List[str],
    ground_truths: List[str],
    top_k: int = 20
) -> Dict[Tuple[str, str], int]:
    """
    Analyze most common character-level confusions.
    
    Returns:
        Dictionary mapping (predicted_char, gt_char) → count
    """
    confusion_counts = {}
    
    for pred, gt in zip(predictions, ground_truths):
        if len(pred) != len(gt):
            continue  # Skip length mismatches
        
        for p_char, g_char in zip(pred, gt):
            if p_char != g_char:
                pair = (p_char, g_char)
                confusion_counts[pair] = confusion_counts.get(pair, 0) + 1
    
    # Sort by frequency
    sorted_pairs = sorted(confusion_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🔀 Top {min(top_k, len(sorted_pairs))} Character Confusions:")
    print("   Pred → GT | Count")
    print("   " + "-" * 25)
    for (pred_c, gt_c), count in sorted_pairs[:top_k]:
        print(f"   {pred_c:>4} → {gt_c:<4} | {count:>4}")
    
    return dict(sorted_pairs[:top_k])


def print_error_analysis(
    predictions: List[str],
    ground_truths: List[str],
    confidences: List[float],
    track_ids: List[str] = None
):
    """Print detailed error analysis statistics."""
    total = len(predictions)
    correct = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    
    # Length errors
    wrong_length = sum(1 for p, g in zip(predictions, ground_truths) 
                      if len(p) != len(g))
    
    # Character-level accuracy (for sequences with correct length)
    char_correct = 0
    char_total = 0
    for p, g in zip(predictions, ground_truths):
        if len(p) == len(g):
            char_correct += sum(1 for pc, gc in zip(p, g) if pc == gc)
            char_total += len(g)
    
    print("\n" + "=" * 60)
    print("  📊 Error Analysis")
    print("=" * 60)
    print(f"  Sequence Accuracy : {correct}/{total} = {100*correct/total:.2f}%")
    print(f"  Wrong Length      : {wrong_length}/{total} = {100*wrong_length/total:.2f}%")
    if char_total > 0:
        print(f"  Char-level Acc    : {char_correct}/{char_total} = {100*char_correct/char_total:.2f}%")
    
    # Confidence statistics
    avg_conf = np.mean(confidences)
    conf_correct = [c for c, p, g in zip(confidences, predictions, ground_truths) if p == g]
    conf_wrong = [c for c, p, g in zip(confidences, predictions, ground_truths) if p != g]
    
    print(f"  Avg Confidence    : {avg_conf:.3f}")
    if conf_correct:
        print(f"    - Correct preds : {np.mean(conf_correct):.3f}")
    if conf_wrong:
        print(f"    - Wrong preds   : {np.mean(conf_wrong):.3f}")
    
    # Show worst cases
    errors = [(p, g, c, tid) for p, g, c, tid in 
              zip(predictions, ground_truths, confidences, track_ids or ['?']*total)
              if p != g]
    if errors:
        errors.sort(key=lambda x: x[2])  # Sort by confidence
        print(f"\n  🔴 Worst {min(5, len(errors))} Errors (lowest confidence):")
        for pred, gt, conf, tid in errors[:5]:
            print(f"     {tid:15s} | Pred: {pred:10s} GT: {gt:10s} | Conf: {conf:.3f}")
    
    print("=" * 60 + "\n")
