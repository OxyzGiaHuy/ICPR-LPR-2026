"""Reusable model components for multi-frame OCR."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet34_Weights, resnet34


class STNBlock(nn.Module):
    """
    Spatial Transformer Network (STN) for image alignment.
    Learns to crop and rectify images before feeding them to the backbone.
    """
    def __init__(self, in_channels: int = 3):
        super().__init__()
        
        # Localization network: Predicts transformation parameters
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d((4, 8)) # Output fixed size for FC
        )
        
        # Regressor for the 3x2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 8, 128),
            nn.ReLU(True),
            nn.Linear(128, 6)
        )
        
        # Initialize the weights/bias with identity transformation
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, C, H, W]
        Returns:
            theta: Affine transformation matrix [Batch, 2, 3]
        """
        xs = self.localization(x)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        return theta


class AttentionFusion(nn.Module):
    """
    Attention-based fusion module for combining multi-frame features.
    Computes a weighted sum of features from multiple frames based on their 'quality' scores.
    """
    def __init__(self, channels: int):
        super().__init__()
        # A small CNN to predict attention scores (quality map) from features
        self.score_net = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 8, 1, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature maps from all frames. Shape: [Batch * Frames, C, H, W]
        Returns:
            Fused feature map. Shape: [Batch, C, H, W]
        """
        total_frames, c, h, w = x.size()
        num_frames = 5  # Fixed based on dataset
        batch_size = total_frames // num_frames

        # Reshape to [Batch, Frames, C, H, W]
        x_view = x.view(batch_size, num_frames, c, h, w)
        
        # Calculate attention scores: [Batch, Frames, 1, H, W]
        scores = self.score_net(x).view(batch_size, num_frames, 1, h, w)
        weights = F.softmax(scores, dim=1)  # Normalize scores across frames

        # Weighted sum fusion
        fused_features = torch.sum(x_view * weights, dim=1)
        return fused_features


class CNNBackbone(nn.Module):
    """A simple CNN backbone for CRNN baseline."""
    def __init__(self, out_channels=512):
        super().__init__()
        # Defined as a list of layers for clarity: Conv -> ReLU -> Pool
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 2
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            # Block 3
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 4
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 2), (2, 1), (0, 1)),
            # Block 5 (Map to sequence height 1)
            nn.Conv2d(512, out_channels, 2, 1, 0), nn.BatchNorm2d(out_channels), nn.ReLU(True)
        )

    def forward(self, x):
        return self.features(x)


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based backbone customized for OCR.
    Uses ResNet34 with modified strides to preserve width (sequence length) while reducing height.
    """
    def __init__(self, pretrained: bool = False):
        super().__init__()
        
        # Load ResNet34 from torchvision
        weights = ResNet34_Weights.DEFAULT if pretrained else None
        resnet = resnet34(weights=weights)

        # --- OCR Customization ---
        # We need to keep the standard first layer (stride 2)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Modify strides in layer3 and layer4 to (2, 1)
        # This reduces height but preserves width for sequence modeling
        self.layer3[0].conv1.stride = (2, 1)
        self.layer3[0].downsample[0].stride = (2, 1)
        
        self.layer4[0].conv1.stride = (2, 1)
        self.layer4[0].downsample[0].stride = (2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input images [Batch, 3, H, W]
        Returns:
            Features [Batch, 512, H // 16, W // 2] (approx)
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Ensure height is 1 for sequence modeling (Height collapsing)
        # Output shape: [Batch, 512, 1, W']
        x = F.adaptive_avg_pool2d(x, (1, None))
        return x


class MoEFeedForward(nn.Module):
    """
    Mixture of Experts Feed-Forward Network.
    
    Replaces the standard FFN in a Transformer layer with N parallel expert FFNs.
    Each token is independently routed to the top-K experts via a learned router.
    Includes an auxiliary load-balancing loss to prevent expert collapse.

    Args:
        d_model:     Input/output dimension (e.g., 512).
        num_experts: Total number of expert FFNs (e.g., 4 or 8).
        top_k:       Number of experts activated per token (e.g., 1 or 2).
        ff_dim:      Inner dimension of each expert FFN (e.g., 2048).
        dropout:     Dropout rate inside each expert.
    """
    def __init__(
        self,
        d_model: int,
        num_experts: int = 4,
        top_k: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert top_k <= num_experts, "top_k must be <= num_experts"
        self.num_experts = num_experts
        self.top_k = top_k

        # Router: maps each token vector to a score over experts
        self.router = nn.Linear(d_model, num_experts, bias=False)

        # Expert pool: each expert is a two-layer FFN (same as standard Transformer FFN)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, ff_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(ff_dim, d_model),
                nn.Dropout(dropout),
            )
            for _ in range(num_experts)
        ])

        # Stores the auxiliary load-balancing loss from the last forward pass.
        # The trainer should add `model.moe_aux_loss()` to the main CTC loss.
        self._aux_loss: torch.Tensor = torch.tensor(0.0)
        # Fraction of tokens routed to each expert (shape [E]), updated each forward
        self._expert_load: torch.Tensor = torch.ones(num_experts) / num_experts

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len, d_model]
        Returns:
            out: [Batch, Seq_Len, d_model]
        """
        B, S, D = x.shape

        # --- 1. Routing ---
        # router_logits: [B*S, num_experts]
        flat_x = x.view(-1, D)
        router_logits = self.router(flat_x)
        router_probs = F.softmax(router_logits, dim=-1)  # [B*S, E]

        # Select top-K experts per token
        top_k_weights, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # [B*S, K]
        # Re-normalise the selected weights so they sum to 1
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)

        # --- 2. Compute auxiliary load-balancing loss (encourage uniform usage) ---
        # f_i = fraction of tokens routed to expert i (top-1 assignment)
        # p_i = mean routing probability for expert i
        # L_aux = num_experts * Σ f_i * p_i  (from Switch Transformer)
        with torch.no_grad():
            # top-1 hard assignment for f_i
            top1_indices = router_probs.argmax(dim=-1)  # [B*S]
            one_hot = F.one_hot(top1_indices, num_classes=self.num_experts).float()
            f = one_hot.mean(dim=0)  # [E], fraction of tokens per expert
        p = router_probs.mean(dim=0)  # [E], mean probability per expert
        self._aux_loss = self.num_experts * (f * p).sum()
        self._expert_load = f.detach().cpu()  # save for diagnostics

        # --- 3. Expert computation ---
        # Process each expert only on the tokens assigned to it (sparse dispatch)
        output = torch.zeros_like(flat_x)  # [B*S, D]

        for expert_idx, expert in enumerate(self.experts):
            # Find tokens where this expert is in the top-K
            # top_k_indices: [B*S, K]
            mask = (top_k_indices == expert_idx)  # [B*S, K] bool
            token_mask = mask.any(dim=-1)          # [B*S] bool

            if not token_mask.any():
                continue

            selected_tokens = flat_x[token_mask]          # [T, D]
            expert_out = expert(selected_tokens)            # [T, D]

            # Gather the weight this expert has for each selected token
            # Shape: [T] – picks the weight from the correct top-K slot
            expert_weights = top_k_weights[token_mask][mask[token_mask]]  # [T]

            output[token_mask] += expert_weights.unsqueeze(-1) * expert_out

        return output.view(B, S, D)

    def get_aux_loss(self) -> torch.Tensor:
        """Returns the cached auxiliary load-balancing loss."""
        return self._aux_loss

    def get_expert_load(self) -> torch.Tensor:
        """Returns the fraction of tokens routed to each expert (last batch)."""
        return self._expert_load


class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer with MoE-FFN replacing the standard FFN.
    Mirrors the interface of nn.TransformerEncoderLayer (batch_first=True).

    Args:
        d_model, nhead, dropout: Standard Transformer parameters.
        num_experts, top_k, ff_dim: MoE-FFN parameters.
    """
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int = 4,
        top_k: int = 2,
        ff_dim: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Multi-Head Self-Attention (identical to standard layer)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

        # MoE FFN instead of standard Linear→GELU→Linear
        self.moe_ffn = MoEFeedForward(
            d_model=d_model,
            num_experts=num_experts,
            top_k=top_k,
            ff_dim=ff_dim,
            dropout=dropout,
        )

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        src_mask=None,
        src_key_padding_mask=None,
        **kwargs,  # absorb extra kwargs from nn.TransformerEncoder internals
    ) -> torch.Tensor:
        # Pre-LN variant (more stable training)
        # 1. Self-Attention sub-layer
        residual = src
        src2, _ = self.self_attn(
            self.norm1(src),
            self.norm1(src),
            self.norm1(src),
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )
        src = residual + self.dropout(src2)

        # 2. MoE-FFN sub-layer
        residual = src
        src = residual + self.moe_ffn(self.norm2(src))
        return src

    def get_aux_loss(self) -> torch.Tensor:
        return self.moe_ffn.get_aux_loss()

    def get_expert_load(self) -> torch.Tensor:
        return self.moe_ffn.get_expert_load()


class PositionalEncoding(nn.Module):
    """
    Injects information about the relative or absolute position of the tokens in the sequence.
    Standard Sinusoidal implementation from 'Attention Is All You Need'.
    """
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input sequence [Batch, Seq_Len, Dim]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)