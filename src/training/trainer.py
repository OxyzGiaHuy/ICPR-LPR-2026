"""Trainer class encapsulating the training and validation loop."""
import csv
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.utils.common import seed_everything
from src.utils.postprocess import decode_with_confidence
from src.utils.visualize import visualize_errors, analyze_confusion_pairs, print_error_analysis
from src.utils.character_balance import (
    compute_character_weights,
    weighted_ctc_loss,
    print_weight_statistics,
    compute_char_frequencies_from_dataset
)


class Trainer:
    """Encapsulates training, validation, and inference logic."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        config,
        idx2char: Dict[int, str]
    ):
        """
        Args:
            model: The neural network model.
            train_loader: Training data loader.
            val_loader: Validation data loader (can be None).
            config: Configuration object with training parameters.
            idx2char: Index to character mapping for decoding.
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.idx2char = idx2char
        self.device = config.DEVICE
        seed_everything(config.SEED, benchmark=config.USE_CUDNN_BENCHMARK)
        
        # Character frequency balancing (addresses 7.5x imbalance from EDA)
        # Compute frequencies from actual training data (with caching)
        if config.USE_CHARACTER_BALANCING:
            print("\n" + "="*60)
            print("⚖️  CHARACTER FREQUENCY BALANCING")
            print("="*60)
            
            # Compute/load character frequencies from dataset
            cache_path = os.path.join("data", ".cache", "char_frequencies.json")
            char_frequencies = compute_char_frequencies_from_dataset(
                train_loader.dataset,
                cache_path=cache_path
            )
            
            # Compute weights
            self.char_weights = compute_character_weights(
                config.CHAR2IDX,
                frequencies=char_frequencies,
                smoothing=config.CHAR_WEIGHT_SMOOTHING
            ).to(self.device)
            
            print(f"Smoothing factor: {config.CHAR_WEIGHT_SMOOTHING}")
            print_weight_statistics(self.char_weights, config.CHAR2IDX, idx2char)
        else:
            # Fallback: use uniform weights
            self.char_weights = None
        
        # Loss and optimizer
        self.criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='mean')  # Fallback
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.LEARNING_RATE,
            weight_decay=config.WEIGHT_DECAY
        )
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.LEARNING_RATE,
            steps_per_epoch=len(train_loader),
            epochs=config.EPOCHS
        )
        self.scaler = GradScaler()
        
        # Tracking
        self.best_acc = 0.0
        self.current_epoch = 0
    
    def _get_output_path(self, filename: str) -> str:
        """Get full path for output file in configured directory."""
        output_dir = getattr(self.config, 'OUTPUT_DIR', 'results')
        os.makedirs(output_dir, exist_ok=True)
        return os.path.join(output_dir, filename)
    
    def _get_exp_name(self) -> str:
        """Get experiment name from config."""
        return getattr(self.config, 'EXPERIMENT_NAME', 'baseline')

    def _get_expert_load(self):
        """
        Collect per-expert load fractions from all MoE layers.
        Returns a list of tensors (one per layer), each shape [num_experts].
        Only meaningful right after a forward pass.
        """
        loads = []
        for layer in getattr(self.model, 'moe_layers', []):
            ffn = getattr(layer, 'moe_ffn', None)
            if ffn is not None and hasattr(ffn, '_aux_loss'):
                # Reconstruct load from the router probs stored during forward
                # (we store them explicitly below via monkey-patch approach)
                pass
        return loads

    def train_one_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        epoch_loss    = 0.0
        epoch_ctc     = 0.0
        epoch_aux     = 0.0
        has_moe = hasattr(self.model, 'get_moe_aux_loss')

        pbar = tqdm(self.train_loader, desc=f"Ep {self.current_epoch + 1}/{self.config.EPOCHS}")
        
        for images, targets, target_lengths, _, _ in pbar:
            images = images.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            with autocast('cuda'):
                preds = self.model(images)
                preds_permuted = preds.permute(1, 0, 2)
                input_lengths = torch.full(
                    size=(images.size(0),),
                    fill_value=preds.size(1),
                    dtype=torch.long
                )
                
                # Use weighted CTC loss if character balancing enabled
                if self.config.USE_CHARACTER_BALANCING:
                    ctc_loss = weighted_ctc_loss(
                        preds_permuted, targets, input_lengths, target_lengths,
                        self.char_weights, blank=0, reduction='mean', zero_infinity=True
                    )
                else:
                    ctc_loss = self.criterion(preds_permuted, targets, input_lengths, target_lengths)
                if has_moe:
                    aux_loss = self.model.get_moe_aux_loss()
                    loss = ctc_loss + aux_loss
                else:
                    aux_loss = torch.tensor(0.0)
                    loss = ctc_loss

            # Scale loss & backward
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.GRAD_CLIP)
            scale_before = self.scaler.get_scale()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scaler.get_scale() >= scale_before:
                self.scheduler.step()
            
            epoch_loss += loss.item()
            epoch_ctc  += ctc_loss.item()
            epoch_aux  += aux_loss.item()

            postfix = {
                'loss': f'{loss.item():.3f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}',
            }
            if has_moe:
                postfix['ctc'] = f'{ctc_loss.item():.3f}'
                postfix['aux'] = f'{aux_loss.item():.4f}'
            pbar.set_postfix(postfix)
        
        n = len(self.train_loader)
        # Store for epoch-level logging
        self._last_epoch_ctc = epoch_ctc / n
        self._last_epoch_aux = epoch_aux / n
        return epoch_loss / n

    def validate(self) -> Tuple[Dict[str, float], List[str]]:
        """Run validation and generate submission data.
        
        Returns:
            Tuple of (metrics_dict, submission_data).
            metrics_dict contains at least 'loss' and 'acc'.
        """
        if self.val_loader is None:
            return {'loss': 0.0, 'acc': 0.0, 'cer': 0.0}, []
        
        self.model.eval()
        val_loss = 0.0
        total_correct = 0
        total_samples = 0
        all_preds: List[str] = []
        all_targets: List[str] = []
        all_confs: List[float] = []
        all_track_ids: List[str] = []
        submission_data: List[str] = []
        
        # Store first batch for visualization
        first_batch_images = None
        first_batch_preds = []
        first_batch_gts = []
        first_batch_confs = []
        first_batch_ids = []
        
        with torch.no_grad():
            for batch_idx, (images, targets, target_lengths, labels_text, track_ids) in enumerate(self.val_loader):
                images = images.to(self.device)
                targets = targets.to(self.device)
                preds = self.model(images)
                
                input_lengths = torch.full(
                    (images.size(0),),
                    preds.size(1),
                    dtype=torch.long
                )
                loss = self.criterion(
                    preds.permute(1, 0, 2),
                    targets,
                    input_lengths,
                    target_lengths
                )
                val_loss += loss.item()

                # Decode predictions
                decoded_list = decode_with_confidence(preds, self.idx2char)

                for i, (pred_text, conf) in enumerate(decoded_list):
                    gt_text = labels_text[i]
                    track_id = track_ids[i]
                    
                    all_preds.append(pred_text)
                    all_targets.append(gt_text)
                    all_confs.append(conf)
                    all_track_ids.append(track_id)
                    
                    if pred_text == gt_text:
                        total_correct += 1
                    submission_data.append(f"{track_id},{pred_text};{conf:.4f}")
                    
                total_samples += len(labels_text)
                
                # Save first batch for visualization
                if batch_idx == 0:
                    first_batch_images = images.cpu()
                    first_batch_preds = [p for p, _ in decoded_list]
                    first_batch_gts = list(labels_text)
                    first_batch_confs = [c for _, c in decoded_list]
                    first_batch_ids = list(track_ids)

        avg_val_loss = val_loss / len(self.val_loader)
        val_acc = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0
        
        metrics = {
            'loss': avg_val_loss,
            'acc': val_acc,
            'preds': all_preds,
            'targets': all_targets,
            'confs': all_confs,
            'track_ids': all_track_ids,
            'first_batch': {
                'images': first_batch_images,
                'preds': first_batch_preds,
                'gts': first_batch_gts,
                'confs': first_batch_confs,
                'ids': first_batch_ids,
            }
        }
        
        return metrics, submission_data

    def save_submission(self, submission_data: List[str]) -> None:
        """Save submission file with experiment name."""
        exp_name = self._get_exp_name()
        filename = self._get_output_path(f"submission_{exp_name}.txt")
        with open(filename, 'w') as f:
            f.write("\n".join(submission_data))
        print(f"📝 Saved {len(submission_data)} lines to {filename}")

    def save_model(self, path: str = None) -> None:
        """Save model checkpoint with experiment name."""
        if path is None:
            exp_name = self._get_exp_name()
            path = self._get_output_path(f"{exp_name}_best.pth")
        torch.save(self.model.state_dict(), path)

    def plot_history(self, log_path: str) -> None:
        """Read the CSV training log and save loss + accuracy plots."""
        try:
            import matplotlib
            matplotlib.use('Agg')  # non-interactive backend (works on Kaggle)
            import matplotlib.pyplot as plt
        except ImportError:
            print("  ⚠️  matplotlib not available – skipping plot.")
            return

        epochs, train_loss, ctc_loss, aux_loss, val_loss, val_acc = [], [], [], [], [], []
        with open(log_path, newline='') as f:
            for row in csv.DictReader(f):
                epochs.append(int(row['epoch']))
                train_loss.append(float(row['train_loss']))
                ctc_loss.append(float(row['ctc_loss']))
                aux_loss.append(float(row['aux_loss']))
                val_loss.append(float(row['val_loss']))
                val_acc.append(float(row['val_acc']))

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(f"Training History — {self._get_exp_name()}", fontsize=13)

        # --- Loss subplot ---
        ax = axes[0]
        ax.plot(epochs, train_loss, label='Train Loss', linewidth=1.5)
        ax.plot(epochs, val_loss,   label='Val Loss',   linewidth=1.5)
        if any(v > 0 for v in ctc_loss):
            ax.plot(epochs, ctc_loss, label='CTC Loss', linestyle='--', linewidth=1)
        if any(v > 0 for v in aux_loss):
            ax.plot(epochs, aux_loss, label='Aux Loss', linestyle=':', linewidth=1)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Loss Curves')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # --- Accuracy subplot ---
        ax = axes[1]
        ax.plot(epochs, val_acc, color='tab:green', linewidth=1.5, label='Val Acc')
        best_epoch = epochs[val_acc.index(max(val_acc))]
        best_val   = max(val_acc)
        ax.axvline(x=best_epoch, color='red', linestyle='--', alpha=0.5,
                   label=f'Best: {best_val:.2f}% @ ep {best_epoch}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title('Val Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = log_path.replace('.csv', '_plot.png')
        plt.savefig(plot_path, dpi=120, bbox_inches='tight')
        plt.close()
        print(f"  📈 History plot saved → {plot_path}")

    def fit(self) -> None:
        """Run the full training loop for specified number of epochs."""
        print(f"🚀 TRAINING START | Device: {self.device} | Epochs: {self.config.EPOCHS}")
        
        # --- CSV log setup ---
        exp_name = self._get_exp_name()
        log_path = self._get_output_path(f"history_{exp_name}.csv")
        _csv_file   = open(log_path, 'w', newline='')
        _csv_writer = csv.DictWriter(
            _csv_file,
            fieldnames=['epoch', 'train_loss', 'ctc_loss', 'aux_loss',
                        'val_loss', 'val_acc', 'lr']
        )
        _csv_writer.writeheader()
        print(f"  📝 Logging to {log_path}")

        for epoch in range(self.config.EPOCHS):
            self.current_epoch = epoch
            
            # Training
            avg_train_loss = self.train_one_epoch()
            
            # Validation
            val_metrics, submission_data = self.validate()
            val_loss = val_metrics['loss']
            val_acc = val_metrics['acc']
            current_lr = self.scheduler.get_last_lr()[0]
            
            # Log results
            has_moe = hasattr(self.model, 'moe_layers')
            moe_info = ""
            if has_moe:
                ctc_val  = getattr(self, '_last_epoch_ctc', avg_train_loss)
                aux_val  = getattr(self, '_last_epoch_aux', 0.0)
                moe_info = f" | CTC: {ctc_val:.4f} | AuxLoss: {aux_val:.5f}"

            print(f"Epoch {epoch + 1}/{self.config.EPOCHS}: "
                  f"Train Loss: {avg_train_loss:.4f}{moe_info} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.2f}% | "
                  f"LR: {current_lr:.2e}")

            # --- Write CSV row ---
            _csv_writer.writerow({
                'epoch':      epoch + 1,
                'train_loss': round(avg_train_loss, 6),
                'ctc_loss':   round(getattr(self, '_last_epoch_ctc', avg_train_loss), 6),
                'aux_loss':   round(getattr(self, '_last_epoch_aux', 0.0), 6),
                'val_loss':   round(val_loss, 6),
                'val_acc':    round(val_acc, 4),
                'lr':         f'{current_lr:.6e}',
            })
            _csv_file.flush()  # visible immediately on Kaggle

            # Expert load diagnostics every 5 epochs
            if has_moe and (epoch + 1) % 5 == 0:
                # Check whether components.py has been patched with get_expert_load()
                sample_layer = self.model.moe_layers[0]
                if not hasattr(sample_layer, 'get_expert_load'):
                    print("  ⚠️  [MoE diag] get_expert_load() missing on MoETransformerEncoderLayer.")
                    print("      → Run the '%%writefile src/models/components.py' patch cell and restart.")
                else:
                    loads = []
                    for layer in self.model.moe_layers:
                        load = layer.get_expert_load()   # [E] cpu tensor
                        if load is None:
                            print(f"  ⚠️  [MoE diag] layer.get_expert_load() returned None – "
                                  "forward pass may not have run yet.")
                            break
                        loads.append(load)
                    if loads and len(loads) == len(self.model.moe_layers):
                        avg_load = torch.stack(loads).mean(dim=0)  # [E]
                        load_str = "  ".join([f"E{i}:{v:.2f}" for i, v in enumerate(avg_load.tolist())])
                        ideal = 1.0 / avg_load.numel()
                        max_imbalance = (avg_load - ideal).abs().max().item()
                        print(f"  📊 Expert Load (avg across layers): [{load_str}]")
                        print(f"     Ideal={ideal:.2f} | Max imbalance={max_imbalance:.3f}"
                              + (" ✅" if max_imbalance < 0.15 else " ⚠️  experts unbalanced"))
            
            # Save best model
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.save_model()
                exp_name = self._get_exp_name()
                model_path = self._get_output_path(f"{exp_name}_best.pth")
                print(f"  ⭐ Saved Best Model: {model_path} ({val_acc:.2f}%)")
                
                if submission_data:
                    self.save_submission(submission_data)
                
                # Visualize errors at best checkpoint
                first_batch = val_metrics.get('first_batch')
                if first_batch and first_batch['images'] is not None:
                    vis_path = self._get_output_path(f"errors_best_{exp_name}.png")
                    visualize_errors(
                        images=first_batch['images'],
                        predictions=first_batch['preds'],
                        ground_truths=first_batch['gts'],
                        confidences=first_batch['confs'],
                        track_ids=first_batch['ids'],
                        output_path=vis_path,
                        max_samples=32,
                        sort_by="wrong_first"
                    )
        
        # Save final model if no validation was performed (submission mode)
        if self.val_loader is None:
            self.save_model()
            exp_name = self._get_exp_name()
            model_path = self._get_output_path(f"{exp_name}_best.pth")
            print(f"  💾 Saved final model: {model_path}")
        
        _csv_file.close()
        print(f"\n✅ Training complete! Best Val Acc: {self.best_acc:.2f}%")
        print(f"  📄 Full log → {log_path}")
        self.plot_history(log_path)
        
        # Final error analysis on full validation set
        print("\n" + "="*60)
        print("  🔍 Running Final Error Analysis...")
        print("="*60)
        final_metrics, _ = self.validate()
        if 'preds' in final_metrics:
            print_error_analysis(
                predictions=final_metrics['preds'],
                ground_truths=final_metrics['targets'],
                confidences=final_metrics['confs'],
                track_ids=final_metrics['track_ids']
            )
            analyze_confusion_pairs(
                predictions=final_metrics['preds'],
                ground_truths=final_metrics['targets'],
                top_k=15
            )

    def predict(self, loader: DataLoader) -> List[Tuple[str, str, float]]:
        """Run inference on a data loader.
        
        Returns:
            List of (track_id, predicted_text, confidence) tuples.
        """
        self.model.eval()
        results: List[Tuple[str, str, float]] = []
        
        with torch.no_grad():
            for images, _, _, _, track_ids in loader:
                images = images.to(self.device)
                preds = self.model(images)
                
                decoded_list = decode_with_confidence(preds, self.idx2char)
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        return results

    def predict_test(self, test_loader: DataLoader, output_filename: str = "submission_final.txt") -> None:
        """Run inference on test data and save submission file.
        
        Args:
            test_loader: DataLoader for test data.
            output_filename: Name of the submission file to save.
        """
        print(f"🔮 Running inference on test data...")
        
        # Use existing predict method
        results = []
        self.model.eval()
        with torch.no_grad():
            for images, _, _, _, track_ids in tqdm(test_loader, desc="Test Inference"):
                images = images.to(self.device)
                preds = self.model(images)
                decoded_list = decode_with_confidence(preds, self.idx2char)
                
                for i, (pred_text, conf) in enumerate(decoded_list):
                    results.append((track_ids[i], pred_text, conf))
        
        # Format and save submission file
        submission_data = [f"{track_id},{pred_text};{conf:.4f}" for track_id, pred_text, conf in results]
        output_path = self._get_output_path(output_filename)
        with open(output_path, 'w') as f:
            f.write("\n".join(submission_data))
        
        print(f"✅ Saved {len(submission_data)} predictions to {output_path}")
