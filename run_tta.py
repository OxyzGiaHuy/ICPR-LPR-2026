"""Run inference with Test-Time Augmentation on validation set."""
import torch
from torch.utils.data import DataLoader
import argparse

from configs.config import get_default_config
from src.data.dataset import MultiFrameDataset
from src.models.restran import ResTranMoE
from src.utils.tta import validate_with_tta
from src.utils.postprocess import decode_with_confidence
from src.utils.visualize import print_error_analysis


def main():
    parser = argparse.ArgumentParser(description="Run TTA inference")
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--augments', type=int, default=5, help='Number of TTA augmentations (1-5)')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for inference')
    args = parser.parse_args()
    
    # Load config
    config = get_default_config()
    config.BATCH_SIZE = args.batch_size
    
    print(f"🔍 TTA Inference with {args.augments} augmentations")
    print(f"   Checkpoint: {args.checkpoint}")
    
    # Load validation dataset
    val_ds = MultiFrameDataset(
        root_dir=config.DATA_ROOT,
        mode='val',
        split_ratio=config.SPLIT_RATIO,
        img_height=config.IMG_HEIGHT,
        img_width=config.IMG_WIDTH,
        char2idx=config.CHAR2IDX,
        val_split_file=config.VAL_SPLIT_FILE,
        seed=config.SEED,
        augmentation_level='full',
        config=config,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        collate_fn=MultiFrameDataset.collate_fn,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )
    
    print(f"✅ Loaded {len(val_ds)} validation samples")
    
    # Load model
    model = ResTranMoE(
        num_classes=config.NUM_CLASSES,
        num_experts=config.MOE_NUM_EXPERTS,
        top_k=config.MOE_TOP_K,
        aux_loss_weight=config.MOE_AUX_LOSS_WEIGHT,
        transformer_heads=config.TRANSFORMER_HEADS,
        transformer_layers=config.TRANSFORMER_LAYERS,
        transformer_dropout=config.TRANSFORMER_DROPOUT,
        use_stn=config.USE_STN
    ).to(config.DEVICE)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=config.DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    print(f"✅ Loaded model from {args.checkpoint}")
    
    # Setup CTC loss
    criterion = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)
    
    # Run TTA validation
    print(f"\n{'='*60}")
    print(f"  🔬 Running TTA Validation (x{args.augments})")
    print(f"{'='*60}\n")
    
    metrics = validate_with_tta(
        model=model,
        val_loader=val_loader,
        criterion=criterion,
        idx2char=config.IDX2CHAR,
        device=config.DEVICE,
        n_augments=args.augments
    )
    
    # Print results
    print(f"\n{'='*60}")
    print(f"  📊 TTA Results (x{args.augments} augmentations)")
    print(f"{'='*60}")
    print(f"  Validation Accuracy: {metrics['acc']:.2f}%")
    print(f"  Validation Loss: {metrics['loss']:.4f}")
    print(f"  Avg Confidence: {sum(metrics['confidences']) / len(metrics['confidences']):.3f}")
    print(f"{'='*60}\n")
    
    # Error analysis
    if 'preds' in metrics and 'targets' in metrics:
        print_error_analysis(
            predictions=metrics['preds'],
            ground_truths=metrics['targets'],
            confidences=metrics['confidences'],
            track_ids=None
        )


if __name__ == '__main__':
    main()
