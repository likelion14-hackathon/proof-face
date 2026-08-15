"""Phase 2 network: EfficientNet + physics MLP, multitask regression.

Architecture
------------
- **Image backbone**: timm ``efficientnet_b0`` (optionally pretrained), global
  pooled to a feature vector.
- **Physics branch**: an MLP that embeds the Phase-1 feature vector.
- **Fusion**: image + physics features are concatenated.
- **Heads**: three regression heads (pigmentation / erythema / pores) with
  **homoscedastic uncertainty weighting** to balance the multitask loss.
- **Illumination invariance**: a **gradient-reversal layer** feeds a domain
  (illumination-bucket) classifier, so the shared features are pushed to be
  illumination-invariant (domain-adversarial training).

Requires the ``dl`` extra (``uv sync --extra dl``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function

from .dataset import N_ILLUMINATION_BUCKETS, PHYSICS_DIM, TARGET_NAMES


class _GradReverse(Function):
    """Gradient Reversal: identity forward, negated-scaled gradient backward."""

    @staticmethod
    def forward(ctx, x, lambd):  # type: ignore[override]
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):  # type: ignore[override]
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Apply the gradient-reversal layer with strength ``lambd``."""
    return _GradReverse.apply(x, lambd)


class PhysicsMLP(nn.Module):
    """Embed the Phase-1 physics feature vector."""

    def __init__(self, in_dim: int = PHYSICS_DIM, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(inplace=True),
            nn.LayerNorm(64),
            nn.Linear(64, embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _RegressionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class SkinNet(nn.Module):
    """Multitask skin-metric regressor with domain-adversarial invariance.

    Parameters
    ----------
    physics_dim : int, optional
        Dimension of the physics feature vector.
    pretrained : bool, optional
        Load ImageNet-pretrained backbone weights (needs network access). Set
        ``False`` for offline / smoke tests.
    physics_embed : int, optional
        Physics embedding size.
    n_illumination : int, optional
        Number of illumination buckets for the domain classifier.
    backbone_name : str, optional
        timm model name for the image backbone.

    Notes
    -----
    ``forward`` returns a dict with ``predictions`` ``(B, 3)`` (order
    :data:`~skin_metrics.models.dataset.TARGET_NAMES`) and ``domain_logits``
    ``(B, n_illumination)``. The learnable ``log_vars`` parameter holds the
    per-task log-variance used by :func:`uncertainty_weighted_loss`.
    """

    def __init__(
        self,
        physics_dim: int = PHYSICS_DIM,
        pretrained: bool = False,
        physics_embed: int = 32,
        n_illumination: int = N_ILLUMINATION_BUCKETS,
        backbone_name: str = "efficientnet_b0",
    ):
        super().__init__()
        import timm

        self.backbone = timm.create_model(
            backbone_name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        feat_dim = self.backbone.num_features
        self.physics_mlp = PhysicsMLP(physics_dim, physics_embed)
        fused_dim = feat_dim + physics_embed

        self.n_tasks = len(TARGET_NAMES)
        self.heads = nn.ModuleList([_RegressionHead(fused_dim) for _ in range(self.n_tasks)])

        # Homoscedastic uncertainty weights (log variance per task).
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))

        # Domain-adversarial illumination classifier (fed through GRL).
        self.domain_classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, n_illumination),
        )

    def forward(self, image: torch.Tensor, physics: torch.Tensor, grl_lambda: float = 1.0) -> dict:
        img_feat = self.backbone(image)
        phys_feat = self.physics_mlp(physics)
        fused = torch.cat([img_feat, phys_feat], dim=1)

        preds = torch.stack([head(fused) for head in self.heads], dim=1)  # (B, 3)
        domain_logits = self.domain_classifier(grad_reverse(fused, grl_lambda))
        return {"predictions": preds, "domain_logits": domain_logits}


def uncertainty_weighted_loss(
    per_task_loss: torch.Tensor, log_vars: torch.Tensor
) -> torch.Tensor:
    """Combine per-task losses with homoscedastic uncertainty weighting.

    ``L = sum_i [ exp(-s_i) * L_i + s_i ]`` where ``s_i = log_vars[i]``
    (Kendall et al., 2018). Tasks with higher learned uncertainty are
    down-weighted automatically.

    Parameters
    ----------
    per_task_loss : torch.Tensor
        Shape ``(n_tasks,)`` of scalar per-task losses.
    log_vars : torch.Tensor
        Shape ``(n_tasks,)`` learnable log-variances.

    Returns
    -------
    torch.Tensor
        Scalar combined loss.
    """
    precision = torch.exp(-log_vars)
    return torch.sum(precision * per_task_loss + log_vars)
