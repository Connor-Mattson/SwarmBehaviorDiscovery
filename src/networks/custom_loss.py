import torch
from typing import Callable, Optional
from torch._prims_common import (
    Tensor,
    TensorLikeType,
)

def _apply_loss_reduction(loss: TensorLikeType, reduction: str) -> TensorLikeType:
    if reduction == "sum":
        return torch.sum(loss)
    elif reduction == "mean":
        return torch.mean(loss)
    else:  # reduction == "none"
        return loss


def _check_reduction_value(reduction: str):
    if reduction not in ("mean", "sum", "none"):
        raise ValueError(f"{reduction} is not a valid value for reduction")

def _triplet_margin_loss(
    anchor: TensorLikeType,
    positive: TensorLikeType,
    negative: TensorLikeType,
    *,
    margin: float = 1.0,
    reduction: str = "mean",
) -> TensorLikeType:
    _check_reduction_value(reduction)

    a_dim = anchor.ndim
    p_dim = positive.ndim
    n_dim = negative.ndim
    assert a_dim == p_dim and p_dim == n_dim

    dist_pos = torch.pairwise_distance(anchor, positive)
    dist_neg = torch.pairwise_distance(anchor, negative)
    mean_neg = torch.mean(dist_neg)
    loss = torch.clamp_min(margin + ((dist_pos - dist_neg) / mean_neg), 0)
    return _apply_loss_reduction(loss, reduction)

class CustomTripletMarginLoss(torch.nn.Module):
    __constants__ = ['margin', 'p', 'eps', 'swap', 'reduction']
    margin: float
    p: float
    eps: float
    swap: bool

    def __init__(self, margin: float = 1.0, p: float = 2., eps: float = 1e-6, swap: bool = False, size_average=None,
                 reduce=None, reduction: str = 'mean'):

        super().__init__()
        self.margin = margin
        self.p = p
        self.eps = eps
        self.swap = swap
        self.reduction = reduction

    def forward(self, anchor: Tensor, positive: Tensor, negative: Tensor) -> Tensor:
        return _triplet_margin_loss(anchor, positive, negative, margin=self.margin, reduction=self.reduction)