"""ResTranOCR: ResNet34 + Transformer architecture (Advanced) with STN."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.components import (
    ResNetFeatureExtractor, AttentionFusion, PositionalEncoding, STNBlock,
    MoETransformerEncoderLayer,
)

class ResTranOCR(nn.Module):
    """
    Modern OCR architecture using optional STN, ResNet34 and Transformer.
    Pipeline: Input (5 frames) -> [Optional STN] -> ResNet34 -> Attention Fusion -> Transformer -> CTC Head
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        
        # 1. Spatial Transformer Network
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone: ResNet34
        self.backbone = ResNetFeatureExtractor(pretrained=False)
        
        # 3. Attention Fusion
        self.fusion = AttentionFusion(channels=self.cnn_channels)
        
        # 4. Transformer Encoder
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cnn_channels,
            nhead=transformer_heads,
            dim_feedforward=transformer_ff_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # 5. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            Logits: [Batch, Seq_Len, Num_Classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)  # [B*F, C, H, W]
        
        if self.use_stn:
            theta = self.stn(x_flat)  # [B*F, 2, 3]
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_aligned = F.grid_sample(x_flat, grid, align_corners=False)
        else:
            x_aligned = x_flat
        
        features = self.backbone(x_aligned)  # [B*F, 512, 1, W']
        fused = self.fusion(features)       # [B, 512, 1, W']
        
        # Prepare for Transformer: [B, C, 1, W'] -> [B, W', C]
        seq_input = fused.squeeze(2).permute(0, 2, 1)
        
        # Add Positional Encoding and pass through Transformer
        seq_input = self.pos_encoder(seq_input)
        seq_out = self.transformer(seq_input) # [B, W', C]
        
        out = self.head(seq_out)              # [B, W', Num_Classes]
        return out.log_softmax(2)


class ResTranMoE(nn.Module):
    """
    ResTranOCR with Mixture of Experts (MoE) replacing the Transformer FFN layers.

    Architecture:
        Input [B, 5, 3, H, W]
        └─ STN (optional)
        └─ ResNet34 backbone → [B*5, 512, 1, W']
        └─ AttentionFusion   → [B,   512, 1, W']
        └─ PositionalEncoding
        └─ MoETransformerEncoderLayer × N  ← key upgrade
        └─ Linear Head → log_softmax → CTC

    Each MoETransformerEncoderLayer replaces the standard FFN with `num_experts`
    parallel expert FFNs, dispatching each sequence token to the top-K most
    relevant experts (as scored by a learned router).

    Domain motivation (LP recognition):
        - Character positions in Brazilian/Mercosur plates have strong positional
          biases (letters at pos 1-3, digits at pos 4-7).
        - MoE experts can specialise per character class / position without
          increasing inference cost proportionally.
        - Scenario-A (controlled) vs Scenario-B (rain/night) may activate
          different expert subsets, improving generalisation across conditions.

    Args:
        num_classes:     Vocab size (chars + 1 blank for CTC).
        transformer_heads:   Number of attention heads.
        transformer_layers:  Number of MoE transformer layers.
        transformer_ff_dim:  Inner dim of each expert FFN.
        dropout:         Dropout rate.
        use_stn:         Whether to prepend the Spatial Transformer Network.
        num_experts:     Total experts per layer (recommended: 4 or 8).
        moe_top_k:       Experts activated per token (recommended: 1 or 2).
        moe_aux_loss_weight: Weight λ for load-balancing loss added to CTC loss.
    """
    def __init__(
        self,
        num_classes: int,
        transformer_heads: int = 8,
        transformer_layers: int = 3,
        transformer_ff_dim: int = 2048,
        dropout: float = 0.1,
        use_stn: bool = True,
        num_experts: int = 4,
        moe_top_k: int = 2,
        moe_aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.cnn_channels = 512
        self.use_stn = use_stn
        self.moe_aux_loss_weight = moe_aux_loss_weight

        # 1. Spatial Transformer Network (optional)
        if self.use_stn:
            self.stn = STNBlock(in_channels=3)

        # 2. Backbone
        self.backbone = ResNetFeatureExtractor(pretrained=False)

        # 3. Attention Fusion (unchanged)
        self.fusion = AttentionFusion(channels=self.cnn_channels)

        # 4. Positional Encoding (unchanged)
        self.pos_encoder = PositionalEncoding(d_model=self.cnn_channels, dropout=dropout)

        # 5. MoE Transformer Encoder (replaces nn.TransformerEncoderLayer stack)
        self.moe_layers = nn.ModuleList([
            MoETransformerEncoderLayer(
                d_model=self.cnn_channels,
                nhead=transformer_heads,
                num_experts=num_experts,
                top_k=moe_top_k,
                ff_dim=transformer_ff_dim,
                dropout=dropout,
            )
            for _ in range(transformer_layers)
        ])

        # Final LayerNorm (common in Pre-LN Transformer stacks)
        self.final_norm = nn.LayerNorm(self.cnn_channels)

        # 6. Prediction Head
        self.head = nn.Linear(self.cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Frames, 3, H, W]
        Returns:
            log_probs: [Batch, Seq_Len, Num_Classes]
        """
        b, f, c, h, w = x.size()
        x_flat = x.view(b * f, c, h, w)

        if self.use_stn:
            theta = self.stn(x_flat)
            grid = F.affine_grid(theta, x_flat.size(), align_corners=False)
            x_flat = F.grid_sample(x_flat, grid, align_corners=False)

        features = self.backbone(x_flat)  # [B*F, 512, 1, W']
        fused = self.fusion(features)     # [B,   512, 1, W']

        seq = fused.squeeze(2).permute(0, 2, 1)  # [B, W', 512]
        seq = self.pos_encoder(seq)

        for layer in self.moe_layers:
            seq = layer(seq)

        seq = self.final_norm(seq)
        out = self.head(seq)              # [B, W', Num_Classes]
        return out.log_softmax(2)

    def get_moe_aux_loss(self) -> torch.Tensor:
        """
        Returns the weighted sum of load-balancing losses from all MoE layers.
        Call this in the training loop and add to the CTC loss:
            loss = ctc_loss + model.get_moe_aux_loss()
        """
        aux = sum(layer.get_aux_loss() for layer in self.moe_layers)
        return self.moe_aux_loss_weight * aux
