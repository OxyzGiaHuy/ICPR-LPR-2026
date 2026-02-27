"""
Character frequency balancing utilities.

Based on EDA: Letter 'A' appears 7500 times, 'K' only 1000 times (7.5x imbalance).
This module provides weighted CTC loss to give equal learning attention to rare characters.
"""

import os
import json
import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from collections import Counter


# Fallback character frequencies from EDA (16,946 unique plates)
# Used only if auto-computation fails or cache not found
CHAR_FREQUENCIES_FALLBACK = {
    # Letters (sorted by frequency)
    'A': 7500, 'B': 3500, 'C': 3200, 'E': 3900, 'D': 3300,
    'F': 3000, 'G': 2300, 'H': 2900, 'I': 2500, 'J': 2300,
    'K': 1000, 'L': 1100, 'M': 1200, 'N': 1100, 'O': 1400,
    'P': 2000, 'Q': 2600, 'R': 2600, 'S': 2700, 'T': 2300,
    'U': 1300, 'V': 1200, 'W': 1200, 'X': 1500, 'Y': 1600, 'Z': 1500,
    
    # Digits (relatively balanced)
    '0': 5100, '1': 5600, '2': 5700, '3': 5700, '4': 5600,
    '5': 5700, '6': 5500, '7': 5800, '8': 5400, '9': 5400,
}


def compute_char_frequencies_from_dataset(dataset, cache_path: Optional[str] = None) -> Dict[str, int]:
    """
    Compute character frequencies from dataset (with caching).
    
    Args:
        dataset: PyTorch dataset with samples containing 'label' field
        cache_path: Path to save/load cached frequencies (optional)
    
    Returns:
        frequencies: Dictionary mapping character -> count
    """
    # Try to load from cache
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, 'r') as f:
                frequencies = json.load(f)
            print(f"  📊 Loaded character frequencies from cache: {cache_path}")
            return frequencies
        except Exception as e:
            print(f"  ⚠️  Cache load failed: {e}. Computing from dataset...")
    
    # Compute from dataset
    print(f"  📊 Computing character frequencies from {len(dataset)} samples...")
    counter = Counter()
    
    for i in range(len(dataset)):
        try:
            sample = dataset.samples[i]  # Access samples list directly
            label = sample.get('label', '')
            if label:
                counter.update(label)
        except Exception:
            continue
    
    frequencies = dict(counter)
    
    # Save to cache
    if cache_path:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, 'w') as f:
                json.dump(frequencies, f, indent=2)
            print(f"  💾 Cached character frequencies to: {cache_path}")
        except Exception as e:
            print(f"  ⚠️  Cache save failed: {e}")
    
    return frequencies


def compute_character_weights(char2idx: Dict[str, int], 
                             frequencies: Optional[Dict[str, int]] = None,
                             smoothing: float = 0.1) -> torch.Tensor:
    """
    Compute inverse frequency weights for balanced CTC loss.
    
    Args:
        char2idx: Character to index mapping (from config)
        frequencies: Character frequency dictionary. If None, uses fallback from EDA
        smoothing: Smoothing factor to avoid extreme weights (0.0 = no smoothing, 1.0 = uniform)
    
    Returns:
        weights: Tensor of shape (num_classes,) with weight for each class
                 weights[0] = 1.0 (for blank/CTC)
                 weights[1:] = inverse frequency weights
    
    Example:
        >>> char_weights = compute_character_weights(config.CHAR2IDX)
        >>> char_weights[config.CHAR2IDX['A']]  # ~0.67 (common char, lower weight)
        >>> char_weights[config.CHAR2IDX['K']]  # ~5.0 (rare char, higher weight)
    """
    # Use fallback if frequencies not provided
    if frequencies is None:
        frequencies = CHAR_FREQUENCIES_FALLBACK
        print("  ℹ️  Using fallback character frequencies from EDA")
    num_classes = len(char2idx) + 1  # +1 for blank
    weights = torch.ones(num_classes)
    
    # Compute total frequency
    total_count = sum(frequencies.values())
    num_chars = len(frequencies)
    
    # Compute inverse frequency weights
    for char, idx in char2idx.items():
        if char in frequencies:
            freq = frequencies[char]
            # Inverse frequency: total / (freq * num_classes)
            inverse_weight = total_count / (freq * num_chars)
            
            # Apply smoothing: weight = (1 - smoothing) * inverse + smoothing * 1.0
            smoothed_weight = (1 - smoothing) * inverse_weight + smoothing * 1.0
            
            weights[idx] = smoothed_weight
        else:
            # Character not in EDA (shouldn't happen) - use neutral weight
            weights[idx] = 1.0
    
    # Normalize so mean weight = 1.0 (keeps loss magnitude similar)
    weights = weights / weights[1:].mean()  # Exclude blank (index 0)
    
    return weights


def weighted_ctc_loss(log_probs: torch.Tensor,
                     targets: torch.Tensor,
                     input_lengths: torch.Tensor,
                     target_lengths: torch.Tensor,
                     char_weights: torch.Tensor,
                     blank: int = 0,
                     reduction: str = 'mean',
                     zero_infinity: bool = False) -> torch.Tensor:
    """
    Weighted CTC loss that upweights rare characters.
    
    Args:
        log_probs: Log probabilities [T, N, C] (T=time, N=batch, C=classes)
        targets: Target sequences [N, S] or flattened [sum(target_lengths)]
        input_lengths: Lengths of log_probs for each batch [N]
        target_lengths: Lengths of targets for each batch [N]
        char_weights: Character weights [C] from compute_character_weights()
        blank: Blank label index (default 0)
        reduction: 'mean', 'sum', or 'none'
        zero_infinity: Whether to zero infinite losses
    
    Returns:
        loss: Weighted CTC loss (scalar if reduction='mean')
    
    Note:
        This is an approximation. True weighted CTC requires modifying the forward-backward
        algorithm. Here we compute per-example weight based on target character frequencies.
    """
    # Standard CTC loss (per example)
    base_loss = F.ctc_loss(
        log_probs, targets, input_lengths, target_lengths,
        blank=blank, reduction='none', zero_infinity=zero_infinity
    )
    
    # Compute weight for each example based on target characters
    batch_size = base_loss.shape[0]
    example_weights = torch.ones(batch_size, device=log_probs.device)
    
    # Flatten targets if not already
    if targets.dim() == 2:
        # Shape [N, S] -> iterate per example
        for i in range(batch_size):
            target_len = target_lengths[i]
            target_chars = targets[i, :target_len]
            
            # Average weight of characters in this example
            char_weight_sum = sum(char_weights[c].item() for c in target_chars if c != blank)
            example_weights[i] = char_weight_sum / max(target_len.item(), 1)
    else:
        # Shape [sum(target_lengths)] -> need to slice per example
        offset = 0
        for i in range(batch_size):
            target_len = target_lengths[i].item()
            target_chars = targets[offset:offset + target_len]
            
            # Average weight of characters in this example
            char_weight_sum = sum(char_weights[c].item() for c in target_chars if c != blank)
            example_weights[i] = char_weight_sum / max(target_len, 1)
            
            offset += target_len
    
    # Apply weights
    weighted_loss = base_loss * example_weights
    
    if reduction == 'mean':
        return weighted_loss.mean()
    elif reduction == 'sum':
        return weighted_loss.sum()
    else:
        return weighted_loss


def analyze_character_distribution(dataset, char2idx: Dict[str, int]) -> Dict[str, int]:
    """
    Analyze character distribution in a dataset (deprecated - use compute_char_frequencies_from_dataset).
    
    Args:
        dataset: PyTorch dataset with 'text' field
        char2idx: Character to index mapping
    
    Returns:
        frequencies: Dictionary mapping character -> count
    """
    print("  ⚠️  analyze_character_distribution is deprecated. Use compute_char_frequencies_from_dataset instead.")
    return compute_char_frequencies_from_dataset(dataset)


def print_weight_statistics(char_weights: torch.Tensor, 
                           char2idx: Dict[str, int],
                           idx2char: Dict[int, str]):
    """
    Print statistics about character weights.
    
    Args:
        char_weights: Weight tensor from compute_character_weights()
        char2idx: Character to index mapping
        idx2char: Index to character mapping
    """
    print("\n" + "="*60)
    print("Character Frequency Weights")
    print("="*60)
    
    # Get weights for each character
    char_weight_list = []
    for char, idx in char2idx.items():
        weight = char_weights[idx].item()
        freq = CHAR_FREQUENCIES_FALLBACK.get(char, 0)
        char_weight_list.append((char, weight, freq))
    
    # Sort by weight (descending)
    char_weight_list.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Highest Weights (Rare Characters):")
    print(f"{'Char':<6} {'Weight':<8} {'Frequency':<10} {'Note'}")
    print("-" * 60)
    for char, weight, freq in char_weight_list[:10]:
        note = "← Rare, needs attention" if weight > 2.0 else ""
        print(f"{char:<6} {weight:<8.3f} {freq:<10} {note}")
    
    print("\nTop 10 Lowest Weights (Common Characters):")
    print(f"{'Char':<6} {'Weight':<8} {'Frequency':<10} {'Note'}")
    print("-" * 60)
    for char, weight, freq in char_weight_list[-10:]:
        note = "← Common, already learned" if weight < 0.8 else ""
        print(f"{char:<6} {weight:<8.3f} {freq:<10} {note}")
    
    print("\nStatistics:")
    weights_no_blank = char_weights[1:]  # Exclude blank
    print(f"  Mean weight: {weights_no_blank.mean():.3f}")
    print(f"  Std weight:  {weights_no_blank.std():.3f}")
    print(f"  Min weight:  {weights_no_blank.min():.3f} ({idx2char[weights_no_blank.argmin().item() + 1]})")
    print(f"  Max weight:  {weights_no_blank.max():.3f} ({idx2char[weights_no_blank.argmax().item() + 1]})")
    print(f"  Weight range: {weights_no_blank.max() / weights_no_blank.min():.2f}x")
    print("="*60 + "\n")


# Example usage
if __name__ == "__main__":
    # Mock config for testing
    CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    char2idx = {char: idx + 1 for idx, char in enumerate(CHARS)}
    idx2char = {idx + 1: char for idx, char in enumerate(CHARS)}
    
    # Compute weights
    weights = compute_character_weights(char2idx, smoothing=0.1)
    
    # Print statistics
    print_weight_statistics(weights, char2idx, idx2char)
    
    # Example: Compute weighted loss
    print("Example: Weighted CTC Loss")
    print("-" * 60)
    
    # Simulate batch
    T, N, C = 30, 4, len(CHARS) + 1  # time=30, batch=4, classes=37 (36 chars + blank)
    log_probs = torch.randn(T, N, C).log_softmax(dim=-1)
    
    # Two examples: one with common chars, one with rare chars
    targets = torch.tensor([
        [char2idx['A'], char2idx['B'], char2idx['C'], char2idx['1'], char2idx['2'], char2idx['3']],  # Common
        [char2idx['K'], char2idx['L'], char2idx['M'], char2idx['4'], char2idx['5'], char2idx['6']],  # Rare
        [char2idx['A'], char2idx['0'], char2idx['1'], char2idx['2'], char2idx['3'], char2idx['4']],  # Common
        [char2idx['X'], char2idx['Y'], char2idx['Z'], char2idx['7'], char2idx['8'], char2idx['9']],  # Medium
    ])
    input_lengths = torch.tensor([30, 30, 30, 30])
    target_lengths = torch.tensor([6, 6, 6, 6])
    
    # Standard CTC loss
    standard_loss = F.ctc_loss(log_probs, targets, input_lengths, target_lengths, reduction='none')
    
    # Weighted CTC loss
    weighted_loss = weighted_ctc_loss(log_probs, targets, input_lengths, target_lengths, weights, reduction='none')
    
    print(f"\nExample 1 (A,B,C,1,2,3 - common chars):")
    print(f"  Standard loss: {standard_loss[0]:.4f}")
    print(f"  Weighted loss: {weighted_loss[0]:.4f} (ratio: {weighted_loss[0]/standard_loss[0]:.2f}x)")
    
    print(f"\nExample 2 (K,L,M,4,5,6 - rare chars):")
    print(f"  Standard loss: {standard_loss[1]:.4f}")
    print(f"  Weighted loss: {weighted_loss[1]:.4f} (ratio: {weighted_loss[1]/standard_loss[1]:.2f}x) ← Higher weight!")
    
    print(f"\nExample 3 (A,0,1,2,3,4 - common chars):")
    print(f"  Standard loss: {standard_loss[2]:.4f}")
    print(f"  Weighted loss: {weighted_loss[2]:.4f} (ratio: {weighted_loss[2]/standard_loss[2]:.2f}x)")
    
    print(f"\nExample 4 (X,Y,Z,7,8,9 - medium chars):")
    print(f"  Standard loss: {standard_loss[3]:.4f}")
    print(f"  Weighted loss: {weighted_loss[3]:.4f} (ratio: {weighted_loss[3]/standard_loss[3]:.2f}x)")
    
    print("\n✓ Rare characters get higher loss weight → more gradient → better learning!")
