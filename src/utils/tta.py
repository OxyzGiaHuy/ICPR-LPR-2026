"""Test-Time Augmentation for improved inference accuracy."""
import torch
import torch.nn as nn
from typing import Dict, List
import torchvision.transforms.functional as TF


def predict_with_tta(
    model: nn.Module,
    images: torch.Tensor,
    device: str = 'cuda',
    n_augments: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply test-time augmentation to improve prediction robustness.
    
    Strategy:
    1. Original image
    2. Slight brightness adjustment (+10%)
    3. Slight contrast adjustment (+10%)
    4. Horizontal shift (-2px)
    5. Horizontal shift (+2px)
    
    Aggregate predictions by averaging logits before CTC decoding.
    
    Args:
        model: Trained model
        images: Input tensor [N, C, H, W]
        device: Device to run inference
        n_augments: Number of augmentations (default 5)
    
    Returns:
        aggregated_logits: Averaged logits [T, N, C]
        confidence: Mean confidence score
    """
    model.eval()
    all_logits = []
    
    with torch.no_grad():
        # 1. Original
        with torch.amp.autocast('cuda'):
            preds = model(images.to(device))  # [N, T, C]
            all_logits.append(preds)
        
        if n_augments >= 2:
            # 2. Brightness +10%
            aug_imgs = TF.adjust_brightness(images, brightness_factor=1.1)
            with torch.amp.autocast('cuda'):
                preds = model(aug_imgs.to(device))
                all_logits.append(preds)
        
        if n_augments >= 3:
            # 3. Contrast +10%
            aug_imgs = TF.adjust_contrast(images, contrast_factor=1.1)
            with torch.amp.autocast('cuda'):
                preds = model(aug_imgs.to(device))
                all_logits.append(preds)
        
        if n_augments >= 4:
            # 4. Horizontal shift -2px (pad right)
            aug_imgs = torch.nn.functional.pad(images, (0, 2, 0, 0))[:, :, :, 2:]
            with torch.amp.autocast('cuda'):
                preds = model(aug_imgs.to(device))
                all_logits.append(preds)
        
        if n_augments >= 5:
            # 5. Horizontal shift +2px (pad left)
            aug_imgs = torch.nn.functional.pad(images, (2, 0, 0, 0))[:, :, :, :-2]
            with torch.amp.autocast('cuda'):
                preds = model(aug_imgs.to(device))
                all_logits.append(preds)
    
    # Average logits (more stable than averaging probabilities)
    aggregated_logits = torch.stack(all_logits).mean(dim=0)  # [N, T, C]
    
    # Compute average confidence (max prob per timestep)
    probs = torch.softmax(aggregated_logits, dim=-1)
    max_probs, _ = probs.max(dim=-1)  # [N, T]
    confidence = max_probs.mean(dim=-1)  # [N]
    
    return aggregated_logits, confidence


def validate_with_tta(
    model: nn.Module,
    val_loader,
    criterion,
    idx2char: Dict[int, str],
    device: str = 'cuda',
    n_augments: int = 5
) -> Dict:
    """
    Run validation with TTA enabled.
    
    Returns:
        metrics: Dict with 'acc', 'loss', 'preds', 'targets', 'confidences'
    """
    from src.utils.postprocess import decode_with_confidence
    from tqdm import tqdm
    
    model.eval()
    total_correct = 0
    total_samples = 0
    all_preds = []
    all_targets = []
    all_confs = []
    val_loss = 0.0
    
    pbar = tqdm(val_loader, desc=f"Val (TTA x{n_augments})")
    
    for images, targets, target_lengths, track_ids, _ in pbar:
        images = images.to(device)
        targets = targets.to(device)
        
        # Apply TTA
        aggregated_logits, _ = predict_with_tta(
            model, images, device, n_augments
        )
        
        # Compute loss on aggregated logits
        preds_permuted = aggregated_logits.permute(1, 0, 2)  # [T, N, C]
        input_lengths = torch.full(
            size=(images.size(0),),
            fill_value=aggregated_logits.size(1),
            dtype=torch.long
        )
        
        with torch.amp.autocast('cuda'):
            loss = criterion(preds_permuted, targets, input_lengths, target_lengths)
        val_loss += loss.item()
        
        # Decode predictions
        for i in range(images.size(0)):
            pred_str, conf = decode_with_confidence(
                aggregated_logits[i].cpu(),
                idx2char,
                blank=0
            )
            
            # Ground truth
            target_seq = targets[i, :target_lengths[i]].cpu().tolist()
            target_str = ''.join([idx2char.get(idx, '') for idx in target_seq])
            
            all_preds.append(pred_str)
            all_targets.append(target_str)
            all_confs.append(conf)
            
            if pred_str == target_str:
                total_correct += 1
            total_samples += 1
        
        pbar.set_postfix({
            'acc': f'{100.0 * total_correct / total_samples:.2f}%',
            'loss': f'{loss.item():.4f}'
        })
    
    accuracy = 100.0 * total_correct / total_samples
    avg_loss = val_loss / len(val_loader)
    
    return {
        'acc': accuracy,
        'loss': avg_loss,
        'preds': all_preds,
        'targets': all_targets,
        'confidences': all_confs
    }
