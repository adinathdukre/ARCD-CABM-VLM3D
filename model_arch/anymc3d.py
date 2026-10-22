"""
AnyMC3D Backbone — AnyMC3D with two learnable vision blocks on top of DINOv2
Based on: "Revisiting 2D Foundation Models for Scalable 3D Medical Image Classification"
         Liu et al., 2025 (arXiv:2512.12887)

         + "DINOv2 Meets Text: A Unified Framework for Image- and Pixel-Level
            Vision-Language Alignment" Jose et al., 2024 (arXiv:2412.16334)

Changes from v1 (anymc3d.py):
  [v2] Two learnable transformer blocks (VisionBlocks) are added on top of the
       frozen DINOv2 backbone, following dino.txt Section 3.1.

Changes from v2 (anymc3d_backbone.py):
  [v3 - NEW] use_patch_concat flag:
       When True, the mean-pooled patch tokens are concatenated with the
       adapted CLS token to form a 2D-dimensional slice embedding instead
       of D. This doubles the slice-level feature richness at the cost of
       doubling the AttentionPool, fusion, and classifier head dimensions.
       AttentionPool, MultiModalFusion, and the linear head all automatically
       use slice_embed_dim = embed_dim * 2 when this is enabled.

  [v3 - NEW] use_25d flag:
       When True, each slice is represented as a triplet of consecutive
       slices [s-1, s, s+1] stacked along the channel axis (2.5D input).
       Boundary slices are padded by repeating the first/last slice.
       This replaces the single-slice-replicated-to-RGB approach.
       When False (default), the original single-slice replication is used.

Everything else is identical to v2:
  - Single-slice input (one slice replicated to 3 channels) [when use_25d=False]
  - CLS token only as slice embedding [when use_patch_concat=False]
  - Same attention pooling, multi-modal fusion, and classifier head

Adapted for Meningioma Molecular Subtype Classification (MG1-MG4)
using T1-contrast and T2-weighted MRI data.

Usage:
    # Default (v2 behaviour)
    model = AnyMC3D(num_classes=4, modalities=['t1c'])

    # With both new features enabled
    model = AnyMC3D(
        num_classes=4,
        modalities=['t1c'],
        use_patch_concat=True,
        use_25d=True,
    )
"""

import copy
import math
import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from peft import LoraConfig, get_peft_model
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

if os.environ.get("ANYMC3D_INFERENCE_ONLY") == "1":
    # The challenge container instantiates AnyMC3D directly. Avoid importing
    # the Lightning/metrics training stack in that deliberately thin image.
    L = SimpleNamespace(LightningModule=nn.Module)
    AUROC = Accuracy = F1Score = None
    BalancedAccuracy = None
else:
    import lightning as L
    from torchmetrics import AUROC, Accuracy, F1Score
    from balanced_accuracy import BalancedAccuracy


# ---------------------------------------------------------------
# ImageNet normalization constants (DINOv2 pretraining stats)
# ---------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225])


def normalize_slices(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a batch of RGB slices with ImageNet statistics.
    Args:
        x: [B, 3, H, W] float tensor in range [0, 1]
    Returns:
        [B, 3, H, W] normalized tensor
    """
    mean = IMAGENET_MEAN.to(x.device).view(1, 3, 1, 1)
    std  = IMAGENET_STD.to(x.device).view(1, 3, 1, 1)
    return (x - mean) / std


def asymmetric_multilabel_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma_pos: float = 0.0,
    gamma_neg: float = 4.0,
    clip: float = 0.05,
    element_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Numerically stable asymmetric loss for sparse multi-label targets."""
    targets = targets.float()
    probabilities = torch.sigmoid(logits.float())
    positive_probability = probabilities
    negative_probability = 1.0 - probabilities
    if clip > 0:
        negative_probability = (negative_probability + clip).clamp(max=1.0)

    eps = 1.0e-8
    positive_loss = targets * torch.log(positive_probability.clamp_min(eps))
    negative_loss = (1.0 - targets) * torch.log(
        negative_probability.clamp_min(eps)
    )

    positive_focus = (1.0 - positive_probability).pow(float(gamma_pos))
    negative_focus = (1.0 - negative_probability).pow(float(gamma_neg))
    loss = -(positive_focus * positive_loss + negative_focus * negative_loss)

    if element_weights is None:
        return loss.mean()
    weights = element_weights.to(device=loss.device, dtype=loss.dtype)
    return (loss * weights).sum() / weights.sum().clamp_min(1.0)


@torch.no_grad()
def consensus_selective_negative_weights(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets: torch.Tensor,
    threshold: float = 0.9,
    temperature: float = 0.05,
    minimum_weight: float = 0.25,
) -> torch.Tensor:
    """Downweight only negative labels contradicted by both image views.

    Positive labels always retain weight one. For a released negative label,
    the smaller positive probability from two independently augmented or
    reconstructed views controls a smooth reliability gate. Requiring both
    views to cross the confidence threshold is deliberately conservative.
    """
    if not 0.0 < temperature:
        raise ValueError("temperature must be positive")
    if not 0.0 <= minimum_weight <= 1.0:
        raise ValueError("minimum_weight must be in [0, 1]")
    targets = targets.float()
    consensus = torch.minimum(
        torch.sigmoid(logits_a.float()),
        torch.sigmoid(logits_b.float()),
    )
    reliable_negative = torch.sigmoid(
        (float(threshold) - consensus) / float(temperature)
    )
    negative_weight = minimum_weight + (1.0 - minimum_weight) * reliable_negative
    return targets + (1.0 - targets) * negative_weight


def tail_aware_class_parameters(
    positive_counts: list[int] | tuple[int, ...],
    max_margin: float = 0.5,
    effective_beta: float = 0.9999,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build LDAM margins and effective-number positive weights.

    The largest margin is assigned to the rarest finding. Effective-number
    weights are mean-normalized so enabling deferred reweighting does not
    abruptly change the overall loss scale.
    """
    counts = torch.as_tensor(positive_counts, dtype=torch.float32)
    if counts.ndim != 1 or counts.numel() == 0:
        raise ValueError("positive_counts must be a non-empty one-dimensional list")
    if torch.any(counts <= 0):
        raise ValueError("positive_counts must contain only positive values")
    if max_margin < 0:
        raise ValueError("max_margin must be non-negative")
    if not 0.0 <= effective_beta < 1.0:
        raise ValueError("effective_beta must be in [0, 1)")

    raw_margins = counts.pow(-0.25)
    margins = float(max_margin) * raw_margins / raw_margins.max()
    if effective_beta == 0.0:
        weights = torch.ones_like(counts)
    else:
        effective_number = 1.0 - torch.pow(float(effective_beta), counts)
        weights = (1.0 - float(effective_beta)) / effective_number.clamp_min(1.0e-12)
        weights = weights / weights.mean()
    return margins, weights


def comorbidity_adjusted_positive_weights(
    targets: torch.Tensor,
    class_weights: torch.Tensor,
    strength: float = 0.0,
    maximum_ratio: float = 2.0,
) -> torch.Tensor:
    """Refine tail weights using the positive labels present in each scan.

    Frequency-only reweighting can repeatedly count a common abnormality when
    it co-occurs with several rarer findings. For every positive label, this
    function compares its global tail weight with the mean weight of the other
    positives in that scan. Rare findings therefore receive more of the
    positive gradient in comorbid scans, while single-finding scans retain the
    original class-balanced weight. Negative entries always have weight one.
    """
    if not 0.0 <= float(strength) <= 1.0:
        raise ValueError("strength must be in [0, 1]")
    if float(maximum_ratio) < 1.0:
        raise ValueError("maximum_ratio must be at least 1")
    if targets.ndim != 2 or class_weights.ndim != 1:
        raise ValueError("targets must be [B,K] and class_weights must be [K]")
    if targets.shape[1] != class_weights.numel():
        raise ValueError("targets and class_weights disagree on class count")

    targets = targets.float()
    class_weights = class_weights.to(device=targets.device, dtype=targets.dtype)
    base = class_weights.unsqueeze(0).expand_as(targets)
    positive_count = targets.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean_positive_weight = (
        targets * base
    ).sum(dim=1, keepdim=True) / positive_count
    relative = (base / mean_positive_weight.clamp_min(1.0e-8)).clamp(
        min=1.0 / float(maximum_ratio),
        max=float(maximum_ratio),
    )
    adjusted = base * relative.pow(float(strength))
    return 1.0 + targets * (adjusted - 1.0)


class AcquisitionConditionedGate(nn.Module):
    """Choose the informative reconstruction separately for every disease."""

    def __init__(
        self,
        metadata_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        evidence_scale: float = 0.5,
        metadata_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.evidence_scale = float(evidence_scale)
        self.metadata_dropout = float(metadata_dropout)
        self.scorer = nn.Sequential(
            nn.LayerNorm(metadata_dim),
            nn.Linear(metadata_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_classes),
        )
        # Begin as the controlled equal-metadata gate. Image evidence can still
        # distinguish views on step zero; metadata preferences are learned.
        nn.init.zeros_(self.scorer[-1].weight)
        nn.init.zeros_(self.scorer[-1].bias)

    def forward(
        self,
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        metadata_a: torch.Tensor,
        metadata_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        metadata_a = metadata_a.to(device=logits_a.device, dtype=logits_a.dtype)
        metadata_b = metadata_b.to(device=logits_b.device, dtype=logits_b.dtype)
        # Detached image evidence lets the gate recognize a view with a clear
        # finding without creating a shortcut gradient into the classifier.
        metadata_a = F.dropout(
            metadata_a, p=self.metadata_dropout, training=self.training
        )
        metadata_b = F.dropout(
            metadata_b, p=self.metadata_dropout, training=self.training
        )
        score_a = self.scorer(metadata_a) + self.evidence_scale * logits_a.detach()
        score_b = self.scorer(metadata_b) + self.evidence_scale * logits_b.detach()
        weights = torch.softmax(torch.stack((score_a, score_b), dim=-1), dim=-1)
        fused_logits = weights[..., 0] * logits_a + weights[..., 1] * logits_b
        return fused_logits, weights


def acquisition_distillation_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    teacher_logits: torch.Tensor,
    pair_valid: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Distill family evidence into either reconstruction at inference time."""
    if temperature <= 0:
        raise ValueError("distillation temperature must be positive")
    teacher_probability = torch.sigmoid(
        teacher_logits.detach().float() / float(temperature)
    )
    # Ambiguous teacher decisions should not dominate hard-label supervision.
    confidence = (2.0 * (teacher_probability - 0.5).abs()).square()
    loss_a = F.binary_cross_entropy_with_logits(
        logits_a.float() / float(temperature),
        teacher_probability,
        reduction='none',
    )
    loss_b = F.binary_cross_entropy_with_logits(
        logits_b.float() / float(temperature),
        teacher_probability,
        reduction='none',
    )
    valid = pair_valid.float().to(logits_a.device)[:, None]
    weights = confidence * valid
    loss = 0.5 * (loss_a + loss_b) * weights
    return (
        loss.sum() / weights.sum().clamp_min(1.0)
        * float(temperature) ** 2
    )


def positive_view_routing_loss(
    gate_weights: torch.Tensor,
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    targets: torch.Tensor,
    pair_valid: torch.Tensor,
    minimum_margin: float = 0.25,
) -> torch.Tensor:
    """Teach metadata routing only from confident released-positive pairs."""
    difference = (logits_a.detach().float() - logits_b.detach().float())
    preferred_weight = torch.where(
        difference >= 0,
        gate_weights[..., 0].float(),
        gate_weights[..., 1].float(),
    )
    mask = (
        targets.float()
        * (difference.abs() >= float(minimum_margin)).float()
        * pair_valid.float().to(logits_a.device)[:, None]
    )
    return (
        -preferred_weight.clamp_min(1.0e-8).log() * mask
    ).sum() / mask.sum().clamp_min(1.0)


# ---------------------------------------------------------------
# Learnable Vision Blocks
# ---------------------------------------------------------------

class VisionBlock(nn.Module):
    """
    A single standard transformer block: multi-head self-attention + MLP.
    Preserves input dimensionality D throughout — identical to the blocks
    inside DINOv2 but trained from scratch rather than frozen.

    Used to build the two-block adapter ψ from dino.txt Section 3.1.
    Processes the full token sequence [CLS, patch1, ..., patchN] output
    by the frozen backbone, producing an adapted sequence of the same shape.
    """

    def __init__(
        self,
        embed_dim:   int,
        num_heads:   int,
        mlp_ratio:   float = 4.0,
        dropout:     float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = num_heads,
            dropout     = dropout,
            batch_first = True,   # expects [B, seq, D]
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 1+N, D]  full token sequence (CLS + patch tokens)
        Returns:
            x: [B, 1+N, D]  adapted token sequence, same shape
        """
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionBlocks(nn.Module):
    """
    Two stacked VisionBlock layers — the adapter ψ from dino.txt.

    Takes the full token sequence [CLS, f1, ..., fN] from the frozen DINOv2
    backbone and produces an adapted sequence [c', f1', ..., fN'] of the same
    shape.
        ViT-S: D=384,  heads=6
        ViT-B: D=768,  heads=12
        ViT-L: D=1024, heads=16
    """

    _HEAD_MAP = {384: 6, 768: 12, 1024: 16}

    def __init__(
        self,
        embed_dim:    int,
        num_blocks:   int   = 2,
        mlp_ratio:    float = 4.0,
        dropout:      float = 0.0,
    ):
        super().__init__()
        num_heads = self._HEAD_MAP.get(embed_dim, max(1, embed_dim // 64))
        self.blocks = nn.Sequential(*[
            VisionBlock(
                embed_dim  = embed_dim,
                num_heads  = num_heads,
                mlp_ratio  = mlp_ratio,
                dropout    = dropout,
            )
            for _ in range(num_blocks)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, 1+N, D]  CLS token prepended to N patch tokens
        Returns:
                    [B, 1+N, D]  adapted tokens, same shape
        """
        return self.blocks(tokens)


# ---------------------------------------------------------------
# Query-based Attention Pooling (Slice Fusion)
# ---------------------------------------------------------------

class AttentionPool(nn.Module):
    """
    Aggregates a sequence of embeddings using a single learnable query vector.
    Permutation-invariant — suits MRI with variable slice counts.
    Implements Eq. 5 from the AnyMC3D paper.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.query = nn.Parameter(torch.empty(embed_dim))
        nn.init.trunc_normal_(self.query, std=0.02)

    def forward(self, H: torch.Tensor):
        """
        Args:
            H: [B, S, D] sequence of slice embeddings
        Returns:
            v: [B, D]    aggregated volume embedding
            a: [B, S]    attention weights (for heatmap visualization)
        """
        scale  = H.shape[-1] ** 0.5
        scores = torch.einsum('bsd,d->bs', H, self.query) / scale
        a      = F.softmax(scores, dim=-1)
        v      = torch.einsum('bs,bsd->bd', a, H)
        return v, a


class DiseaseAttentionPool(nn.Module):
    """Pool one slice representation per disease instead of one per volume."""

    def __init__(self, embed_dim: int, num_queries: int):
        super().__init__()
        self.queries = nn.Parameter(torch.empty(num_queries, embed_dim))
        nn.init.trunc_normal_(self.queries, std=0.02)

    def forward(self, H: torch.Tensor):
        """Return disease embeddings [B,K,D] and slice attention [B,K,S]."""
        scale = H.shape[-1] ** 0.5
        scores = torch.einsum('bsi,ki->bks', H, self.queries) / scale
        a = F.softmax(scores, dim=-1)
        z = torch.einsum('bks,bsi->bki', a, H)
        return z, a


class DiseaseSpecificClassifier(nn.Module):
    """Independent linear classifier for each disease-specific embedding."""

    def __init__(self, embed_dim: int, num_classes: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.weight = nn.Parameter(torch.empty(num_classes, embed_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.trunc_normal_(self.weight, std=0.02)

    def forward(self, disease_embeddings: torch.Tensor) -> torch.Tensor:
        disease_embeddings = self.dropout(disease_embeddings)
        return torch.einsum(
            'bkd,kd->bk', disease_embeddings, self.weight
        ) + self.bias


class CrossDiseaseRelationAdapter(nn.Module):
    """Residual cross-disease reasoning with near-identity initialization.

    Disease queries localize evidence independently. This adapter lets their
    volume-level embeddings exchange information before classification, which
    captures clinically meaningful co-occurrence and exclusion relationships.
    LayerScale starts at 1e-4 so a warm-start initially preserves ADQ outputs.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        bottleneck_ratio: int = 4,
        dropout: float = 0.1,
        init_scale: float = 1.0e-4,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}"
            )
        hidden_dim = max(embed_dim // bottleneck_ratio, 64)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.attention_scale = nn.Parameter(
            torch.full((embed_dim,), float(init_scale))
        )
        self.mlp_scale = nn.Parameter(
            torch.full((embed_dim,), float(init_scale))
        )

    def forward(self, disease_embeddings: torch.Tensor):
        normalized = self.norm1(disease_embeddings)
        messages, attention = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False,
        )
        refined = disease_embeddings + self.attention_scale * messages
        refined = refined + self.mlp_scale * self.mlp(self.norm2(refined))
        return refined, attention


# ---------------------------------------------------------------
# Single-Modality Encoder
# ---------------------------------------------------------------

class ModalityEncoder(nn.Module):
    """
    Encodes a single 3D MRI volume into a fixed-size embedding.

    New flags vs v2:
      use_patch_concat (bool):
        If True, the slice embedding is [CLS ; mean(patch tokens)], giving a
        2D-dimensional vector. AttentionPool is sized accordingly.
        If False (default), only the adapted CLS token is used (v2 behaviour).

      use_25d (bool):
        If True, each slice is encoded with its two neighbours as a 3-channel
        2.5D input [s-1, s, s+1]. Boundary slices are padded by repeating.
        If False (default), a single slice is replicated to 3 channels (v2).
    """

    def __init__(
        self,
        backbone:           nn.Module,
        embed_dim:          int,
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        input_size:         int   = 224,
        slice_axis:         int   = 3,
        vision_blocks:      int   = 2,
        mlp_ratio:          float = 4.0,
        block_dropout:      float = 0.0,
        # AnyMC3D feature
        use_slice_attn_pool: bool = True,   # True: AttentionPool over slices

        # [Optional features]
        use_patch_concat:   bool  = False,
        use_25d:            bool  = False,
        use_patch_attn_pool: bool = False,   # True: AttentionPool over patches
                                              # False: mean pooling (default)
        num_disease_queries: int = 0,

    ):
        super().__init__()
        self.embed_dim            = embed_dim
        self.input_size           = input_size
        self.slice_axis           = slice_axis
        self.use_slice_attn_pool  = use_slice_attn_pool
        self.use_patch_concat     = use_patch_concat
        self.use_25d              = use_25d
        self.use_patch_attn_pool  = use_patch_attn_pool
        self.num_disease_queries  = int(num_disease_queries)

        # Slice embedding dimensionality: doubled when patch tokens are concat'd
        self.slice_embed_dim = embed_dim * 2 if use_patch_concat else embed_dim

        lora_config = LoraConfig(
            r              = lora_rank,
            lora_alpha     = lora_alpha,
            target_modules = ["qkv", "proj", "patch_embed.proj"],
            lora_dropout   = 0.0,
            bias           = "none",
        )
        self.adapted_backbone = get_peft_model(backbone, lora_config)

        if vision_blocks > 0:
            self.vision_blocks = VisionBlocks(
                embed_dim  = embed_dim,
                num_blocks = vision_blocks,
                mlp_ratio  = mlp_ratio,
                dropout    = block_dropout,
            )

        if use_slice_attn_pool:
            if self.num_disease_queries > 0:
                self.pool = DiseaseAttentionPool(
                    self.slice_embed_dim, self.num_disease_queries
                )
            else:
                self.pool = AttentionPool(self.slice_embed_dim)

        # Optional learned patch pooling (only instantiated when both flags are active)
        if use_patch_concat and use_patch_attn_pool:
            self.patch_pool = AttentionPool(embed_dim)

    def encode_slices(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract per-slice embeddings from the adapted backbone.

        v2:  returns adapted CLS token c'                    -> [B, D]
        [use_patch_concat=True, use_patch_attn_pool=False]:
             returns [c' ; mean(f1',...,fN')]                -> [B, 2D]
        [use_patch_concat=True, use_patch_attn_pool=True]:
             returns [c' ; AttentionPool(f1',...,fN')]       -> [B, 2D]
        """
        out = self.adapted_backbone.forward_features(x)

        # Reconstruct full token sequence [CLS, patch1, ..., patchN]
        if isinstance(out, dict):
            cls_token    = out['x_norm_clstoken']       # [B, D]
            patch_tokens = out['x_norm_patchtokens']    # [B, N, D]
            tokens = torch.cat(
                [cls_token.unsqueeze(1), patch_tokens], dim=1
            )                                           # [B, 1+N, D]
        else:
            tokens = out                                # [B, 1+N, D]

        # Pass full token sequence through two learnable vision blocks
        if hasattr(self, 'vision_blocks'):
            tokens = self.vision_blocks(tokens)             # [B, 1+N, D]

        # Extract adapted CLS token c' from position 0
        cls_adapted = tokens[:, 0]                      # [B, D]

        if not self.use_patch_concat:
            return cls_adapted                          # [B, D]

        # Pool adapted patch tokens, then concatenate with CLS -> [B, 2D]
        patch_tokens_adapted = tokens[:, 1:]            # [B, N, D]
        if self.use_patch_attn_pool:
            patch_repr, _ = self.patch_pool(patch_tokens_adapted)  # [B, D]
        else:
            patch_repr = patch_tokens_adapted.mean(dim=1)          # [B, D]

        return torch.cat([cls_adapted, patch_repr], dim=-1)        # [B, 2D]

    def _build_25d_input(self, x_perm: torch.Tensor) -> torch.Tensor:
        """
        [NEW v3] Build 2.5D input by stacking each slice with its neighbours.

        Args:
            x_perm: [B, n_slices, C, H, W]  C=1 for single-channel MRI
        Returns:
            flat:   [B*n_slices, 3, H, W]   3-channel triplet per slice
        """
        B, S, C, H, W = x_perm.shape

        # Pad at boundaries by repeating the first and last slice
        x_padded = torch.cat(
            [x_perm[:, :1], x_perm, x_perm[:, -1:]], dim=1
        )                                               # [B, S+2, C, H, W]

        # Stack triplets [s-1, s, s+1] along channel axis
        prev_slices = x_padded[:, :-2]                  # [B, S, C, H, W]
        curr_slices = x_padded[:, 1:-1]                 # [B, S, C, H, W]
        next_slices = x_padded[:, 2:]                   # [B, S, C, H, W]

        triplets = torch.cat(
            [prev_slices, curr_slices, next_slices], dim=2
        )                                               # [B, S, 3C, H, W]
        # Since C=1 for medical images, 3C=3 — exactly the RGB format expected.

        return triplets.reshape(B * S, 3 * C, H, W)    # [B*S, 3, H, W]

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: [B, 1, H, W, S]  values in [0, 1]
        Returns:
            v:    [B, slice_embed_dim]  modality embedding
            attn: [B, n_slices]         attention weights
        """
        B, C, H, W, S = x.shape

        # Move slice axis to position 1 so we can index along it
        spatial_to_full = {1: 2, 2: 3, 3: 4}
        full_axis = spatial_to_full[self.slice_axis]

        perm     = [0, full_axis] + [i for i in range(1, 5) if i != full_axis]
        x_perm   = x.permute(*perm)          # (B, n_slices, C, h, w)
        n_slices = x_perm.shape[1]
        h_sz     = x_perm.shape[3]
        w_sz     = x_perm.shape[4]

        if self.use_25d:
            # [NEW v3] Stack [s-1, s, s+1] as a 3-channel 2.5D input.
            # _build_25d_input handles the boundary padding and reshaping.
            flat = self._build_25d_input(x_perm)        # (B*n, 3, h, w)
        else:
            # Flatten batch+slice. Single-channel volumes are replicated for
            # the RGB-pretrained backbone; CT-RATE already supplies three
            # clinically meaningful CT-window channels.
            flat = x_perm.reshape(B * n_slices, C, h_sz, w_sz)
            if C == 1:
                flat = flat.repeat(1, 3, 1, 1)
            elif C != 3:
                raise ValueError(f"Expected 1 or 3 input channels, got {C}")

        # Resize in-plane to DINOv2 input size
        if h_sz != self.input_size or w_sz != self.input_size:
            flat = F.interpolate(
                flat,
                size=(self.input_size, self.input_size),
                mode='bilinear',
                align_corners=False,
            )

        flat = flat.clamp(0, 1)
        flat = normalize_slices(flat)         # ImageNet stats

        # Chunk DINOv2 forward passes to avoid OOM
        _chunk = int(os.environ.get("ANYMC3D_SLICE_CHUNK", "4"))
        if _chunk < 1:
            raise ValueError("ANYMC3D_SLICE_CHUNK must be a positive integer")
        if flat.shape[0] > _chunk:
            embeddings = torch.cat(
                [self.encode_slices(c) for c in flat.split(_chunk, dim=0)],
                dim=0,
            )
        else:
            embeddings = self.encode_slices(flat)       # (B*n, slice_embed_dim)

        H_seq = rearrange(embeddings, '(b s) d -> b s d', b=B)  # (B, S, D)

        if self.use_slice_attn_pool:
            v, a = self.pool(H_seq)                          # paper method
        else:
            v = H_seq.mean(dim=1)                            # mean pooling ablation
            a = torch.ones(B, H_seq.shape[1],
                        device=H_seq.device) / H_seq.shape[1]  # uniform weights
        return v, a


# ---------------------------------------------------------------
# Multi-Modal Fusion
# ---------------------------------------------------------------

class MultiModalFusion(nn.Module):
    """
    Fuses embeddings from multiple modalities (T1c + T2w) via a second
    stage of attention pooling with a shared learnable task query.
    """

    def __init__(self, embed_dim: int):
        super().__init__()
        self.pool = AttentionPool(embed_dim)

    def forward(self, modality_embeddings: list) -> torch.Tensor:
        """
        Args:
            modality_embeddings: list of [B, D] tensors, one per modality
        Returns:
            [B, D] fused patient-level embedding
        """
        stacked = torch.stack(modality_embeddings, dim=1)  # [B, M, D]
        v, _    = self.pool(stacked)
        return v


# ---------------------------------------------------------------
# Full AnyMC3D Backbone Model
# ---------------------------------------------------------------

class AnyMC3D(nn.Module):
    """
    AnyMC3D Backbone with optional CLS+patch concat and 2.5D slice input.

    Architecture:
      Frozen DINOv2 + LoRA  ->  VisionBlocks (2x, trainable)
      ->  [CLS ; mean(patches)] or CLS only
      ->  AttentionPool (slice fusion)
      ->  [MultiModalFusion]
      ->  Linear head

    New flags:
      use_patch_concat: if True, slice embedding = [CLS ; mean_patch], dim=2D
      use_25d:          if True, input = [s-1, s, s+1] triplet (2.5D)

    Trainable parameters:
      - LoRA adapters inside DINOv2      (~0.5M for ViT-B)
      - VisionBlocks (2 transformer blocks) (~14M for ViT-B, D=768)
      - AttentionPool query              (~768 or ~1536 params)
      - Fusion query (if multi-modal)    (~768 or ~1536 params)
      - Linear classifier head
    """

    def __init__(
        self,
        num_classes:        int   = 4,
        modalities:         list  = ['t1c'],
        backbone_name:      str   = 'dinov2_vitb14',
        lora_rank:          int   = 8,
        lora_alpha:         int   = 16,
        input_size:         int   = 224,
        cls_head_dropout:   float = 0.1,
        slice_axis:         int   = 3,
        vision_blocks:      int   = 2,
        mlp_ratio:          float = 4.0,
        block_dropout:      float = 0.0,
        # [NEW v3]
        use_slice_attn_pool: bool  = True,
        use_patch_concat:   bool  = False,
        use_25d:            bool  = False,
        use_patch_attn_pool: bool = False,
        use_disease_queries: bool = False,
        use_cross_disease_relation: bool = False,
        relation_heads:      int   = 8,
        relation_dropout:    float = 0.1,
    ):
        super().__init__()

        self.num_classes         = num_classes
        self.modalities          = modalities
        self.backbone_name       = backbone_name
        self.use_slice_attn_pool = use_slice_attn_pool
        self.use_patch_concat    = use_patch_concat
        self.use_25d             = use_25d
        self.use_patch_attn_pool = use_patch_attn_pool
        self.use_disease_queries = use_disease_queries
        self.use_cross_disease_relation = use_cross_disease_relation

        if use_disease_queries and len(modalities) != 1:
            raise ValueError(
                "Disease-specific queries currently support one modality only"
            )
        if use_disease_queries and not use_slice_attn_pool:
            raise ValueError(
                "Disease-specific queries require use_slice_attn_pool=True"
            )
        if use_cross_disease_relation and not use_disease_queries:
            raise ValueError(
                "Cross-disease relation reasoning requires disease queries"
            )

        dinov2_repo = os.environ.get("DINOV2_REPO")
        if dinov2_repo:
            print(f"Loading {backbone_name} from local DINOv2 source: {dinov2_repo}")
            # Submission containers run without network access. The trained
            # checkpoint supplies every backbone tensor, so initialize the
            # vendored architecture without downloading pretrained weights.
            backbone = torch.hub.load(
                dinov2_repo,
                backbone_name,
                source="local",
                pretrained=False,
            )
        else:
            print(f"Loading {backbone_name} from torch.hub...")
            backbone = torch.hub.load('facebookresearch/dinov2', backbone_name)
        for param in backbone.parameters():
            param.requires_grad = False

        embed_dim      = backbone.embed_dim
        self.embed_dim = embed_dim
        print(f"Backbone embed_dim: {embed_dim}")

        self.encoders = nn.ModuleDict()
        for modality in modalities:
            self.encoders[modality] = ModalityEncoder(
                backbone             = copy.deepcopy(backbone),
                embed_dim            = embed_dim,
                lora_rank            = lora_rank,
                lora_alpha           = lora_alpha,
                input_size           = input_size,
                slice_axis           = slice_axis,
                vision_blocks        = vision_blocks,
                mlp_ratio            = mlp_ratio,
                block_dropout        = block_dropout,
                use_slice_attn_pool  = use_slice_attn_pool,
                use_patch_concat     = use_patch_concat,
                use_25d              = use_25d,
                use_patch_attn_pool  = use_patch_attn_pool,
                num_disease_queries  = num_classes if use_disease_queries else 0,
            )

        # Slice embedding dimensionality — 2D when patch concat is active
        slice_embed_dim      = embed_dim * 2 if use_patch_concat else embed_dim
        self.slice_embed_dim = slice_embed_dim

        # Multi-modal fusion uses slice_embed_dim throughout
        if len(modalities) > 1:
            self.fusion = MultiModalFusion(slice_embed_dim)

        # Classification head — input is slice_embed_dim
        if use_disease_queries:
            if use_cross_disease_relation:
                self.relation_adapter = CrossDiseaseRelationAdapter(
                    slice_embed_dim,
                    num_heads=relation_heads,
                    dropout=relation_dropout,
                )
            self.classifier = DiseaseSpecificClassifier(
                slice_embed_dim, num_classes, cls_head_dropout
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(cls_head_dropout),
                nn.Linear(slice_embed_dim, num_classes),
            )

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if vision_blocks > 0:
            n_vision_block_params = sum(
                p.numel()
                for enc in self.encoders.values()
                for p in enc.vision_blocks.parameters()
                if p.requires_grad
            )
        print(f"\nAnyMC3D Backbone initialized:")
        print(f"  Modalities:             {modalities}")
        print(f"  Classes:                {num_classes}")
        print(f"  Backbone:               {backbone_name} (embed_dim={embed_dim}, frozen)")
        print(f"  Vision blocks:          {vision_blocks} x transformer block (trainable)")
        print(f"  Vision block params:    {n_vision_block_params:,} per modality" if vision_blocks > 0 else "0 (no vision blocks)")
        print(f"  Slice embed dim:        {slice_embed_dim} (patch_concat={use_patch_concat})")
        print(f"  Slice pooling:          {'AttentionPool' if use_slice_attn_pool else 'mean'} (use_slice_attn_pool={use_slice_attn_pool})")
        print(f"  Disease queries:        {num_classes if use_disease_queries else 0}")
        print(f"  Cross-disease relation: {use_cross_disease_relation}")
        print(f"  Patch pooling:          {'AttentionPool' if use_patch_attn_pool else 'mean'} (use_patch_attn_pool={use_patch_attn_pool})")
        print(f"  2.5D input:             {use_25d}")
        print(f"  Total trainable params: {n_trainable:,}")

    def forward(self, inputs: dict):
        """
        Args:
            inputs: dict mapping modality name -> [B, 1, H, W, S] tensor
                    Values should be pre-normalized to [0, 1]
        Returns:
            logits: [B, num_classes]
            aux:    dict with per-modality attention weights for heatmaps
        """
        aux                 = {}
        modality_embeddings = []

        for modality in self.modalities:
            if modality not in inputs:
                raise KeyError(
                    f"Modality '{modality}' not in inputs. Got: {list(inputs.keys())}"
                )
            v, attn = self.encoders[modality](inputs[modality])
            modality_embeddings.append(v)
            aux[f'attn_{modality}'] = attn

        if len(self.modalities) == 1:
            patient_embedding = modality_embeddings[0]
        else:
            patient_embedding = self.fusion(modality_embeddings)

        if self.use_cross_disease_relation:
            patient_embedding, relation_attention = self.relation_adapter(
                patient_embedding
            )
            aux['disease_relation_attention'] = relation_attention

        logits = self.classifier(patient_embedding)
        return logits, aux


# ---------------------------------------------------------------
# PyTorch Lightning Training Module
# ---------------------------------------------------------------

class AnyMC3DLightningModule(L.LightningModule):
    """
    PyTorch Lightning wrapper for AnyMC3D Backbone.

    Handles:
      - Focal loss for class imbalance (gamma=2, alpha=0.25 as per paper)
      - Three-way parameter grouping for optimizer:
          (1) LoRA adapter params       -> lora_lr
          (2) Vision block params       -> vision_lr
          (3) Head params               -> head_lr
      - Linear warmup + cosine annealing LR schedule
      - AUROC, balanced accuracy, F1 (macro) metrics
    """

    def __init__(
        self,
        num_classes:          int   = 4,
        modalities:           list  = ['t1c'],
        task:                 str   = "multiclass",  
        backbone_name:        str   = 'dinov2_vitl14',
        lora_rank:            int   = 8,
        lora_alpha:           int   = 16,
        input_size:           int   = 224,
        cls_head_dropout:     float = 0.1,
        slice_axis:           int   = 3,
        vision_blocks:        int   = 2,
        mlp_ratio:            float = 4.0,
        block_dropout:        float = 0.0,
        # [NEW v3]
        use_slice_attn_pool:  bool  = True,
        use_patch_concat:     bool  = False,
        use_25d:              bool  = False,
        use_patch_attn_pool:  bool  = False,
        use_disease_queries:  bool  = False,
        use_cross_disease_relation: bool = False,
        relation_heads:       int   = 8,
        relation_dropout:     float = 0.1,
        # Optimizer
        lora_lr:              float = 5e-5,
        lora_weight_decay:    float = 1e-5,
        vision_lr:            float = 1e-4,
        vision_weight_decay:  float = 1e-5,
        head_lr:              float = 5e-4,
        head_weight_decay:    float = 1e-4,
        warmup_epochs:        int   = 10,
        max_epochs:           int   = 150,
        lr_scheduler:         str   = "cosine",
        # Focal loss
        focal_gamma:          float = 2.0,
        focal_alpha:          float = 0.25,
        multilabel_loss:      str   = "focal",
        asl_gamma_pos:        float = 0.0,
        asl_gamma_neg:        float = 4.0,
        asl_clip:             float = 0.05,
        crs_threshold:        float = 0.9,
        crs_temperature:      float = 0.05,
        crs_minimum_weight:   float = 0.25,
        acquisition_metadata_dim: int = 13,
        acquisition_gate_hidden_dim: int = 64,
        acquisition_evidence_scale: float = 0.5,
        acquisition_metadata_dropout: float = 0.0,
        acquisition_gate_entropy_weight: float = 1.0e-3,
        acquisition_distillation_weight: float = 0.0,
        acquisition_distillation_temperature: float = 1.0,
        acquisition_routing_weight: float = 0.0,
        acquisition_routing_minimum_margin: float = 0.25,
        tail_positive_counts: list[int] | None = None,
        tail_margin_max: float = 0.0,
        tail_effective_beta: float = 0.9999,
        tail_drw_start_fraction: float = 1.0,
        tail_comorbidity_strength: float = 0.0,
        tail_comorbidity_maximum_ratio: float = 2.0,
        classifier_balance_weight: float = 0.0,
        consistency_weight:   float = 0.0,
        consistency_temperature: float = 1.0,
        # Training hyperparameters
        seed:                 int   = 0,
        precision:            str   = "16-mixed",
        early_stopping_patience: int = 50,
        save_top_k:           int   = 3,
        log_every_n_steps:    int   = 10,
        devices:              int = 1,
        strategy:            str   = "auto",
        num_nodes:           int   = 1,
        sync_batchnorm:      bool  = False,
        #WandB logging
        project:              str   = "AnyMC3D",
        run_name:             str   = "AnyMC3D_Run",
    ):
        super().__init__()
        self.save_hyperparameters()

        assert task in ('multiclass', 'multilabel'), f"Unknown task: {task}"
        self.task = task
        if multilabel_loss not in ('focal', 'asl', 'crs_asl', 'acer_crs'):
            raise ValueError(
                "multilabel_loss must be one of: focal, asl, crs_asl, acer_crs"
            )

        self.model = AnyMC3D(
            num_classes          = num_classes,
            modalities           = modalities,
            backbone_name        = backbone_name,
            lora_rank            = lora_rank,
            lora_alpha           = lora_alpha,
            input_size           = input_size,
            cls_head_dropout     = cls_head_dropout,
            slice_axis           = slice_axis,
            vision_blocks        = vision_blocks,
            mlp_ratio            = mlp_ratio,
            block_dropout        = block_dropout,
            use_slice_attn_pool  = use_slice_attn_pool,
            use_patch_concat     = use_patch_concat,
            use_25d              = use_25d,
            use_patch_attn_pool  = use_patch_attn_pool,
            use_disease_queries  = use_disease_queries,
            use_cross_disease_relation = use_cross_disease_relation,
            relation_heads       = relation_heads,
            relation_dropout     = relation_dropout,
        )

        self.num_classes = num_classes
        self.modalities  = modalities
        if multilabel_loss == 'acer_crs':
            self.acquisition_gate = AcquisitionConditionedGate(
                metadata_dim=int(acquisition_metadata_dim),
                num_classes=num_classes,
                hidden_dim=int(acquisition_gate_hidden_dim),
                evidence_scale=float(acquisition_evidence_scale),
                metadata_dropout=float(acquisition_metadata_dropout),
            )

        if not 0.0 <= float(tail_drw_start_fraction) <= 1.0:
            raise ValueError("tail_drw_start_fraction must be in [0, 1]")
        if not 0.0 <= float(tail_comorbidity_strength) <= 1.0:
            raise ValueError("tail_comorbidity_strength must be in [0, 1]")
        if float(tail_comorbidity_maximum_ratio) < 1.0:
            raise ValueError("tail_comorbidity_maximum_ratio must be at least 1")
        if float(classifier_balance_weight) < 0.0:
            raise ValueError("classifier_balance_weight must be non-negative")
        tail_enabled = float(tail_margin_max) > 0.0
        if tail_enabled:
            if tail_positive_counts is None or len(tail_positive_counts) != num_classes:
                raise ValueError(
                    "tail_positive_counts must contain one count per class when "
                    "tail_margin_max is enabled"
                )
            tail_margins, tail_weights = tail_aware_class_parameters(
                tail_positive_counts,
                max_margin=float(tail_margin_max),
                effective_beta=float(tail_effective_beta),
            )
        else:
            tail_margins = torch.zeros(num_classes, dtype=torch.float32)
            tail_weights = torch.ones(num_classes, dtype=torch.float32)
        self.register_buffer("tail_positive_margins", tail_margins)
        self.register_buffer("tail_positive_weights", tail_weights)

        if task == 'multilabel':
            metric_kwargs = dict(task='multilabel', num_labels=num_classes)
        elif num_classes == 2:
            metric_kwargs = dict(task='binary')
        else:
            metric_kwargs = dict(task='multiclass', num_classes=num_classes)

        # ── Training metrics (mirror val) ─────────────────────────────────────
        self.train_auroc = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.train_acc   = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.train_f1    = F1Score(average='macro', **metric_kwargs)

        if task != 'multilabel':
            self.train_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.val_auroc    = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.val_acc      = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.val_f1       = F1Score(average='macro', **metric_kwargs)

        # BalancedAccuracy is a multiclass concept — skip it for multilabel
        if task != 'multilabel':
            self.val_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_auroc   = AUROC(average='macro', **metric_kwargs) if task == 'multilabel' else AUROC(**metric_kwargs)
        self.test_acc     = Accuracy(average='macro', **metric_kwargs) if task == 'multilabel' else Accuracy(**metric_kwargs)
        self.test_f1      = F1Score(average='macro', **metric_kwargs)
        if task != 'multilabel':
            self.test_bal_acc = BalancedAccuracy(num_classes=num_classes)

        self.test_preds  = []
        self.test_labels = []

    # ------------------------------------------------------------------
    # Focal Loss
    # ------------------------------------------------------------------

    def focal_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Focal loss with two modes:
        - multiclass: softmax-based, targets are class indices [B]
        - multilabel: sigmoid-based, targets are multi-hot floats [B, K]
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        """
        if self.task == 'multilabel':
            # targets must be float for BCE
            targets = targets.float()
            bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
            pt = torch.exp(-bce)  # prob of correct class per element
            alpha = self.hparams.focal_alpha
            alpha_t = alpha * targets + (1.0 - alpha) * (1.0 - targets)
            focal = alpha_t * (1 - pt) ** self.hparams.focal_gamma * bce
            return focal.mean()
        else:
            ce = F.cross_entropy(logits, targets, reduction='none')
            pt = torch.exp(-ce)
            return (self.hparams.focal_alpha * (1 - pt) ** self.hparams.focal_gamma * ce).mean()

    def classification_loss(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        element_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.task == 'multilabel' and self.hparams.multilabel_loss in (
            'asl', 'crs_asl', 'acer_crs'
        ):
            targets = targets.float()
            if self.training and float(self.hparams.tail_margin_max) > 0.0:
                margins = self.tail_positive_margins.to(
                    device=logits.device, dtype=logits.dtype
                )
                logits = logits - targets.to(logits.dtype) * margins[None, :]

                start_epoch = math.ceil(
                    float(self.hparams.tail_drw_start_fraction)
                    * int(self.hparams.max_epochs)
                )
                if int(self.current_epoch) >= start_epoch:
                    class_weights = self.tail_positive_weights.to(
                        device=logits.device, dtype=logits.dtype
                    )
                    positive_weights = comorbidity_adjusted_positive_weights(
                        targets,
                        class_weights,
                        strength=float(self.hparams.tail_comorbidity_strength),
                        maximum_ratio=float(
                            self.hparams.tail_comorbidity_maximum_ratio
                        ),
                    )
                    if element_weights is None:
                        element_weights = positive_weights
                    else:
                        element_weights = element_weights * positive_weights
            return asymmetric_multilabel_loss(
                logits,
                targets,
                gamma_pos=float(self.hparams.asl_gamma_pos),
                gamma_neg=float(self.hparams.asl_gamma_neg),
                clip=float(self.hparams.asl_clip),
                element_weights=element_weights,
            )
        return self.focal_loss(logits, targets)

    # ------------------------------------------------------------------
    # Batch unpacking
    # ------------------------------------------------------------------

    def _unpack_batch(self, batch):
        x, y, *_ = batch
        if isinstance(x, dict):
            inputs = x
        else:
            inputs = {self.modalities[0]: x}

        paired_inputs = None
        pair_valid = None
        metadata_a = None
        metadata_b = None
        if len(batch) >= 5:
            paired_x = batch[3]
            pair_valid = batch[4].bool()
            if isinstance(paired_x, dict):
                paired_inputs = paired_x
            else:
                paired_inputs = {self.modalities[0]: paired_x}
        if len(batch) >= 7:
            metadata_a = batch[5]
            metadata_b = batch[6]
        return inputs, y, paired_inputs, pair_valid, metadata_a, metadata_b

    @staticmethod
    def _bernoulli_symmetric_kl(
        logits_a: torch.Tensor,
        logits_b: torch.Tensor,
        temperature: float,
    ) -> torch.Tensor:
        """Per-sample symmetric KL between independent Bernoulli outputs."""
        # Evaluate the divergence in float32 even under mixed precision;
        # bfloat16 epsilon is too coarse for stable Bernoulli KL terms.
        eps = 1e-6
        p = torch.sigmoid(logits_a.float() / temperature).clamp(eps, 1.0 - eps)
        q = torch.sigmoid(logits_b.float() / temperature).clamp(eps, 1.0 - eps)
        kl_pq = p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()
        kl_qp = q * (q / p).log() + (1.0 - q) * ((1.0 - q) / (1.0 - p)).log()
        return 0.5 * (kl_pq + kl_qp).mean(dim=1)

    def _shared_step(self, batch):
        inputs, y, paired_inputs, pair_valid, metadata_a, metadata_b = self._unpack_batch(batch)
        logits, _ = self.model(inputs)

        # Every DDP rank must enter the same branch. Some CT-RATE families have
        # only one reconstruction, so branching on pair_valid would make only
        # a subset of ranks call the sync_dist logs below, desynchronizing the
        # collective sequence. Singleton families still provide an independently
        # augmented second view; only their KL term is masked out.
        use_pair = (
            self.training
            and (
                self.hparams.consistency_weight > 0
                or self.hparams.multilabel_loss in ('crs_asl', 'acer_crs')
            )
            and paired_inputs is not None
            and pair_valid is not None
        )
        if use_pair:
            paired_logits, _ = self.model(paired_inputs)
            element_weights = None
            fused_logits = 0.5 * (logits + paired_logits)
            gate_weights = None
            if self.hparams.multilabel_loss == 'acer_crs':
                if metadata_a is None or metadata_b is None:
                    raise ValueError(
                        "acer_crs requires paired CT-RATE acquisition metadata"
                    )
                fused_logits, gate_weights = self.acquisition_gate(
                    logits, paired_logits, metadata_a, metadata_b
                )
                consensus_logits = fused_logits
            else:
                consensus_logits = None

            if self.hparams.multilabel_loss in ('crs_asl', 'acer_crs'):
                if consensus_logits is None:
                    element_weights = consensus_selective_negative_weights(
                        logits,
                        paired_logits,
                        y,
                        threshold=float(self.hparams.crs_threshold),
                        temperature=float(self.hparams.crs_temperature),
                        minimum_weight=float(self.hparams.crs_minimum_weight),
                    )
                else:
                    # Present the gated reconstruction evidence twice so the
                    # existing smooth CRS reliability function operates on the
                    # acquisition-selected posterior rather than min(view).
                    element_weights = consensus_selective_negative_weights(
                        consensus_logits,
                        consensus_logits,
                        y,
                        threshold=float(self.hparams.crs_threshold),
                        temperature=float(self.hparams.crs_temperature),
                        minimum_weight=float(self.hparams.crs_minimum_weight),
                    )
                negative_mask = 1.0 - y.float()
                downweight_fraction = (
                    (1.0 - element_weights) * negative_mask
                ).sum() / negative_mask.sum().clamp_min(1.0)
                self.log(
                    'train/selective_negative_downweight',
                    downweight_fraction,
                    on_step=True,
                    on_epoch=True,
                    sync_dist=False,
                )
            loss = self.classification_loss(logits, y, element_weights)
            paired_loss = self.classification_loss(
                paired_logits, y, element_weights
            )
            if self.hparams.multilabel_loss == 'acer_crs':
                fused_loss = self.classification_loss(
                    fused_logits, y, element_weights
                )
            valid = pair_valid.to(logits.device)
            consistency = self._bernoulli_symmetric_kl(
                logits,
                paired_logits,
                float(self.hparams.consistency_temperature),
            )
            consistency = (consistency * valid).sum() / valid.sum().clamp_min(1)
            if self.hparams.multilabel_loss == 'acer_crs':
                # The entropy term prevents early collapse to a single view;
                # its small coefficient allows class-specific specialization.
                gate_entropy = -(
                    gate_weights.float()
                    * gate_weights.float().clamp_min(1.0e-8).log()
                ).sum(dim=-1).mean()
                loss = (
                    0.25 * (loss + paired_loss)
                    + 0.5 * fused_loss
                    + self.hparams.consistency_weight * consistency
                    - self.hparams.acquisition_gate_entropy_weight * gate_entropy
                )
                self.log(
                    'train/acquisition_gate_entropy', gate_entropy,
                    on_step=True, on_epoch=True, sync_dist=False,
                )
                distillation = acquisition_distillation_loss(
                    logits,
                    paired_logits,
                    fused_logits,
                    pair_valid,
                    temperature=float(
                        self.hparams.acquisition_distillation_temperature
                    ),
                )
                routing = positive_view_routing_loss(
                    gate_weights,
                    logits,
                    paired_logits,
                    y,
                    pair_valid,
                    minimum_margin=float(
                        self.hparams.acquisition_routing_minimum_margin
                    ),
                )
                loss = (
                    loss
                    + self.hparams.acquisition_distillation_weight * distillation
                    + self.hparams.acquisition_routing_weight * routing
                )
                self.log(
                    'train/acquisition_distillation', distillation,
                    on_step=True, on_epoch=True, sync_dist=False,
                )
                self.log(
                    'train/acquisition_routing', routing,
                    on_step=True, on_epoch=True, sync_dist=False,
                )
            else:
                loss = (
                    0.5 * (loss + paired_loss)
                    + self.hparams.consistency_weight * consistency
                )
            self.log(
                'train/reconstruction_consistency', consistency,
                on_step=True, on_epoch=True, sync_dist=False,
            )
            self.log(
                'train/paired_fraction', valid.float().mean(),
                on_step=False, on_epoch=True, sync_dist=False,
            )
            logits = fused_logits
        else:
            loss = self.classification_loss(logits, y)
        if self.training and float(self.hparams.classifier_balance_weight) > 0.0:
            classifier = self.model.classifier
            if not isinstance(classifier, DiseaseSpecificClassifier):
                raise TypeError(
                    "classifier weight balancing requires disease-specific queries"
                )
            classifier_norms = classifier.weight.float().norm(dim=1).clamp_min(1e-8)
            log_norms = classifier_norms.log()
            balance_loss = (log_norms - log_norms.mean()).square().mean()
            loss = loss + float(self.hparams.classifier_balance_weight) * balance_loss
            self.log(
                'train/classifier_balance', balance_loss,
                on_step=True, on_epoch=True, sync_dist=False,
            )

        if self.task == 'multilabel':
            probs = torch.sigmoid(logits)
        else:
            probs = F.softmax(logits, dim=1)
        return loss, probs, y

    # ------------------------------------------------------------------
    # Training / Validation / Test steps
    # ------------------------------------------------------------------

    def training_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel' and float(self.hparams.tail_margin_max) > 0.0:
            start_epoch = math.ceil(
                float(self.hparams.tail_drw_start_fraction)
                * int(self.hparams.max_epochs)
            )
            self.log(
                'train/tail_drw_active',
                float(int(self.current_epoch) >= start_epoch),
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )

        if self.task == 'multilabel':
            y_int = y.int()
            self.train_auroc.update(probs, y_int)
            self.train_acc.update(probs, y_int)
            self.train_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.train_auroc.update(auroc_input, y)
            self.train_acc.update(preds, y)
            self.train_f1.update(preds, y)
            self.train_bal_acc.update(preds, y)

        self.log('train/loss', loss, on_step=True, on_epoch=True,
                prog_bar=True, sync_dist=False)
        return loss
    
    def on_train_epoch_end(self):
        # TorchMetrics.compute() already synchronizes metric state across DDP
        # ranks. Do not all-reduce the resulting scalar a second time.
        self.log('train/AUROC',    self.train_auroc.compute(), prog_bar=False)
        self.log('train/accuracy', self.train_acc.compute(),   prog_bar=False)
        self.log('train/F1_macro', self.train_f1.compute(),    prog_bar=False)
        self.train_auroc.reset(); self.train_acc.reset(); self.train_f1.reset()

        if self.task != 'multilabel':
            self.log('train/balanced_accuracy', self.train_bal_acc.compute(),
                    prog_bar=False)
            self.train_bal_acc.reset()

    def validation_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel':
            # torchmetrics multilabel expects int targets [B, K] and float probs [B, K]
            y_int = y.int()
            self.val_auroc.update(probs, y_int)
            self.val_acc.update(probs, y_int)
            self.val_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.val_auroc.update(auroc_input, y)
            self.val_acc.update(preds, y)
            self.val_f1.update(preds, y)
            self.val_bal_acc.update(preds, y)

        self.log('val/loss', loss, on_epoch=True, prog_bar=True, sync_dist=False)

    def on_validation_epoch_end(self):
        self.log('val/AUROC',    self.val_auroc.compute(), prog_bar=True)
        self.log('val/accuracy', self.val_acc.compute(),   prog_bar=True)
        self.log('val/F1_macro', self.val_f1.compute(),    prog_bar=True)
        self.val_auroc.reset(); self.val_acc.reset(); self.val_f1.reset()

        if self.task != 'multilabel':
            self.log('val/balanced_accuracy', self.val_bal_acc.compute(),
                    prog_bar=True)
            self.val_bal_acc.reset()

    def test_step(self, batch, batch_idx):
        loss, probs, y = self._shared_step(batch)

        if self.task == 'multilabel':
            y_int = y.int()
            self.test_auroc.update(probs, y_int)
            self.test_acc.update(probs, y_int)
            self.test_f1.update(probs, y_int)
        else:
            preds = probs.argmax(dim=1)
            auroc_input = probs[:, 1] if self.num_classes == 2 else probs
            self.test_auroc.update(auroc_input, y)
            self.test_acc.update(preds, y)
            self.test_f1.update(preds, y)
            self.test_bal_acc.update(preds, y)

        self.test_preds.append(probs.cpu())
        self.test_labels.append(y.cpu())

    def on_test_epoch_end(self):
        auroc = self.test_auroc.compute()
        acc   = self.test_acc.compute()
        f1    = self.test_f1.compute()

        self.log('test/AUROC',    auroc)
        self.log('test/accuracy', acc)
        self.log('test/F1_macro', f1)

        print(f"\n{'='*50}")
        print(f"Test Results:")
        print(f"  AUROC (macro):     {auroc:.4f}")
        print(f"  Accuracy (macro):  {acc:.4f}")
        print(f"  F1 (macro):        {f1:.4f}")

        if self.task != 'multilabel':
            balanced_acc = self.test_bal_acc.compute()
            self.log('test/balanced_accuracy', balanced_acc)
            print(f"  Balanced Accuracy: {balanced_acc:.4f}")
            self.test_bal_acc.reset()

        print(f"{'='*50}\n")
        self.test_auroc.reset(); self.test_acc.reset(); self.test_f1.reset()

    def predict_step(self, batch, batch_idx):
        inputs, y, _, _, _, _ = self._unpack_batch(batch)
        logits, aux = self.model(inputs)
        return y, logits

    # ------------------------------------------------------------------
    # Optimizer: three-way LR grouping
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        lora_params   = []
        vision_params = []
        head_params   = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if any(k in name for k in [
                'classifier', 'pool.query', 'pool.queries', 'fusion',
                'relation_adapter',
            ]):
                head_params.append(param)
            elif 'vision_blocks' in name:
                vision_params.append(param)
            else:
                lora_params.append(param)

        if hasattr(self, 'acquisition_gate'):
            head_params.extend(self.acquisition_gate.parameters())

        optimizer = torch.optim.AdamW([
            {'params': lora_params,
             'lr': self.hparams.lora_lr,
             'weight_decay': self.hparams.lora_weight_decay},
            {'params': vision_params,
             'lr': self.hparams.vision_lr,
             'weight_decay': self.hparams.vision_weight_decay},
            {'params': head_params,
             'lr': self.hparams.head_lr,
             'weight_decay': self.hparams.head_weight_decay},
        ])

        if self.hparams.lr_scheduler == "constant":
            return optimizer

        schedulers = []
        milestones  = []

        # Keep short smoke/debug runs valid. A one-epoch run cannot spend one
        # full epoch warming up and then create a zero-length cosine schedule.
        max_epochs = max(1, int(self.hparams.max_epochs))
        warmup_epochs = min(
            max(0, int(self.hparams.warmup_epochs)),
            max_epochs - 1,
        )

        if warmup_epochs > 0:
            schedulers.append(
                LinearLR(optimizer, start_factor=0.1, end_factor=1.0,
                         total_iters=warmup_epochs)
            )
            milestones.append(warmup_epochs)

        schedulers.append(
            CosineAnnealingLR(optimizer,
                              T_max=max(1, max_epochs - warmup_epochs),
                              )
        )

        if len(schedulers) == 1:
            scheduler = schedulers[0]
        else:
            scheduler = SequentialLR(optimizer,
                                     schedulers=schedulers,
                                     milestones=milestones)

        return {
            'optimizer':    optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'epoch'},
        }


# ---------------------------------------------------------------
# Quick sanity check — run with: python anymc3d_backbone.py
# ---------------------------------------------------------------

if __name__ == '__main__':
    print("Running AnyMC3D Backbone sanity check...\n")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")

    B, H, W, S = 2, 64, 64, 64

    dummy_inputs = {
        't1c': torch.randn(B, 1, H, W, S).clamp(0, 1).to(device),
    }

    print("=" * 60)
    print("Case 1: default (v2 behaviour — CLS only, replicated slice)")
    model_v2 = AnyMC3D(
        num_classes       = 4,
        modalities        = ['t1c'],
        backbone_name     = 'dinov2_vits14',
        lora_rank         = 8,
        lora_alpha        = 16,
        input_size        = 224,
        vision_blocks     = 2,
        use_patch_concat  = False,
        use_25d           = False,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v2(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")           # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")  # [2, 64]
    print(f"  embed_dim:     {model_v2.embed_dim}")
    print(f"  slice_embed:   {model_v2.slice_embed_dim}")

    print()
    print("=" * 60)
    print("Case 2: use_patch_concat=True, use_25d=True, mean pooling (default)")
    model_v3 = AnyMC3D(
        num_classes          = 4,
        modalities           = ['t1c'],
        backbone_name        = 'dinov2_vits14',
        lora_rank            = 8,
        lora_alpha           = 16,
        input_size           = 224,
        vision_blocks        = 2,
        use_patch_concat     = True,
        use_25d              = True,
        use_patch_attn_pool  = False,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v3(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")              # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")     # [2, 64]
    print(f"  embed_dim:     {model_v3.embed_dim}")        # D (e.g. 384)
    print(f"  slice_embed:   {model_v3.slice_embed_dim}")  # 2D (e.g. 768)

    print()
    print("=" * 60)
    print("Case 3: use_patch_concat=True, use_25d=True, AttentionPool over patches")
    model_v3_attn = AnyMC3D(
        num_classes          = 4,
        modalities           = ['t1c'],
        backbone_name        = 'dinov2_vits14',
        lora_rank            = 8,
        lora_alpha           = 16,
        input_size           = 224,
        vision_blocks        = 2,
        use_patch_concat     = True,
        use_25d              = True,
        use_patch_attn_pool  = True,
    ).to(device)
    with torch.no_grad():
        logits, aux = model_v3_attn(dummy_inputs)
    print(f"  logits shape:  {logits.shape}")              # [2, 4]
    print(f"  attn shape:    {aux['attn_t1c'].shape}")     # [2, 64]
    print(f"  embed_dim:     {model_v3_attn.embed_dim}")
    print(f"  slice_embed:   {model_v3_attn.slice_embed_dim}")  # 2D (e.g. 768)

    print("\nSanity check passed!")
