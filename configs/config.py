"""Configuration dataclass for the training pipeline."""
import os
from dataclasses import dataclass, field
from typing import Dict
import torch

# ---------------------------------------------------------------------------
# Kaggle input root detection
# Adjust KAGGLE_DATASET_SLUG to match your actual dataset slug on Kaggle.
# e.g. if your dataset URL is kaggle.com/datasets/foo/lrlpr2026 → "foo/lrlpr2026"
# The input directory on Kaggle is /kaggle/input/<dataset-name>/
# ---------------------------------------------------------------------------
_KAGGLE_DATASET_NAME = "lrlpr2026"   # ← change if your slug differs
_KAGGLE_INPUT_DIR = f"/kaggle/input/{_KAGGLE_DATASET_NAME}"


@dataclass
class Config:
    """Training configuration with all hyperparameters."""
    
    # Experiment tracking
    MODEL_TYPE: str = "restran"  # "crnn", "restran", or "restran_moe"
    EXPERIMENT_NAME: str = MODEL_TYPE
    AUGMENTATION_LEVEL: str = "full"  # "full" or "light"
    USE_STN: bool = True  # Enable Spatial Transformer Network
    
    # Data paths
    DATA_ROOT: str = "data/train"
    TEST_DATA_ROOT: str = "data/public_test"
    VAL_SPLIT_FILE: str = "data/val_tracks.json"
    SUBMISSION_FILE: str = "submission.txt"
    
    IMG_HEIGHT: int = 32
    IMG_WIDTH: int = 128
    
    # Character set
    CHARS: str = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    
    # Character frequency balancing (EDA-driven optimization)
    # EDA shows 7.5x imbalance: Letter 'A' = 7500 occurrences, 'K' = 1000
    # Weighted CTC loss gives rare characters more gradient → better learning
    USE_CHARACTER_BALANCING: bool = True       # Enable weighted CTC loss
    CHAR_WEIGHT_SMOOTHING: float = 0.1         # 0.0 = pure inverse freq, 1.0 = uniform
    OVERSAMPLE_RARE_CHARS: bool = True         # Duplicate rare char samples 3x
    RARE_CHAR_THRESHOLD: int = 2000           # Characters with freq < 2000 = rare (K,L,M,N,O...)
    
    # Training hyperparameters
    BATCH_SIZE: int = 64
    LEARNING_RATE: float = 5e-4
    EPOCHS: int = 80  # Increased for hard Scenario B val set (30 for quick test, 80 for full training)
    SEED: int = 42
    NUM_WORKERS: int = 4  # Kaggle recommends ≤4 workers to avoid slowdowns
    WEIGHT_DECAY: float = 3e-4  # Stronger regularization for LR images
    GRAD_CLIP: float = 5.0
    SPLIT_RATIO: float = 0.9
    USE_CUDNN_BENCHMARK: bool = False
    
    # Early stopping
    EARLY_STOPPING_PATIENCE: int = 15  # Stop if no improvement for 15 epochs
    
    # Test-time augmentation (TTA)
    USE_TTA: bool = False              # Enable TTA for validation (slower but more accurate)
    TTA_AUGMENTS: int = 5              # Number of augmentations (1=off, 5=full)
    
    # CRNN model hyperparameters
    HIDDEN_SIZE: int = 256
    RNN_DROPOUT: float = 0.25
    
    # ResTranOCR model hyperparameters
    TRANSFORMER_HEADS: int = 8
    TRANSFORMER_LAYERS: int = 3
    TRANSFORMER_FF_DIM: int = 2048
    TRANSFORMER_DROPOUT: float = 0.15  # Higher dropout for LR robustness

    # MoE hyperparameters (only used when MODEL_TYPE = "restran_moe")
    # Design rationale: Val set = 100% Scenario B (degraded) but train = 50% A + 50% B
    # 4 experts for hybrid routing: position (letter/digit) × quality (clear/degraded)
    # - E0: Letters (pos 0-2) + Clear | E1: Letters + Degraded  
    # - E2: Digits (pos 3-6) + Clear  | E3: Digits + Degraded
    # Top-k=2 allows both position expert + quality expert to activate per token
    MOE_NUM_EXPERTS: int = 4           # 4-expert hybrid: position × quality matrix
    MOE_TOP_K: int = 2                 # Hybrid routing: activate 2 experts per token
    MOE_AUX_LOSS_WEIGHT: float = 0.01  # Light load-balancing (routing learns naturally from data)
    
    DEVICE: torch.device = field(default_factory=lambda: torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    OUTPUT_DIR: str = "results"
    
    # Derived attributes (computed in __post_init__)
    CHAR2IDX: Dict[str, int] = field(default_factory=dict, init=False)
    IDX2CHAR: Dict[int, str] = field(default_factory=dict, init=False)
    NUM_CLASSES: int = field(default=0, init=False)
    
    def __post_init__(self):
        """Compute derived attributes after initialization."""
        # Auto-detect Kaggle and reroute data paths so rsync is not needed
        if os.path.isdir(_KAGGLE_INPUT_DIR):
            if self.DATA_ROOT == "data/train":
                self.DATA_ROOT = os.path.join(_KAGGLE_INPUT_DIR, "train")
            if self.TEST_DATA_ROOT == "data/public_test":
                self.TEST_DATA_ROOT = os.path.join(_KAGGLE_INPUT_DIR, "public_test")
            # val_split_file lives inside the cloned repo — keep it relative
            print(f"[Config] Kaggle detected → DATA_ROOT = {self.DATA_ROOT}")

        self.CHAR2IDX = {char: idx + 1 for idx, char in enumerate(self.CHARS)}
        self.IDX2CHAR = {idx + 1: char for idx, char in enumerate(self.CHARS)}
        self.NUM_CLASSES = len(self.CHARS) + 1  # +1 for blank


def get_default_config() -> Config:
    """Returns the default configuration."""
    return Config()
