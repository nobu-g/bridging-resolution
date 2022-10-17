from typing import Dict, Callable

import torch
import torch.nn.functional as F

eps = 1e-6


def cross_entropy_loss(output: torch.Tensor,  # (b, seq, seq)
                       target: torch.Tensor,  # (b, seq, seq)
                       mask: torch.Tensor,  # (b, seq, seq)
                       ) -> torch.Tensor:  # ()
    losses = F.cross_entropy(output, target, reduction='none')  # (b, seq, seq)
    # reduce using masked mean
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)


def binary_cross_entropy_with_logits(output: torch.Tensor,  # (b, seq, seq)
                                     target: torch.Tensor,  # (b, seq, seq)
                                     mask: torch.Tensor,  # (b, seq, seq)
                                     ) -> torch.Tensor:  # ()
    losses = F.binary_cross_entropy_with_logits(output, target, reduction='none')  # (b, seq, seq)
    # reduce using masked mean
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)


def margin_ranking_loss(output: torch.Tensor,  # (b, seq, seq)
                        target: torch.Tensor,  # (b, seq, seq)
                        mask: torch.Tensor  # (b, seq, seq)
                        ) -> torch.Tensor:  # ()
    delta_output = output.unsqueeze(3) - output.unsqueeze(2)  # (b, seq, seq, seq)
    delta_target = target.unsqueeze(3) - target.unsqueeze(2)  # (b, seq, seq, seq)
    # (b, seq, seq, seq)
    losses = torch.max(torch.zeros_like(delta_output), delta_target.sign() * (-delta_output + delta_target))
    mask = torch.tril(mask.unsqueeze(-1).expand_as(delta_output), diagonal=-1)  # (b, seq, seq, seq)
    return torch.sum(losses * mask).div(torch.sum(mask) + eps)


def mse_loss(output: torch.Tensor,  # (b, seq, seq)
             target: torch.Tensor,  # (b, seq, seq)
             mask: torch.Tensor,  # (b, seq, seq)
             ) -> torch.Tensor:  # ()
    error = (output - target) * mask.float()
    squared_error = error * error
    return torch.sum(squared_error).div(torch.sum(mask) + eps)


def mse_total_constraint_loss(output: torch.Tensor,  # (b, seq, seq)
                              target: torch.Tensor,  # (b, seq, seq)
                              mask: torch.Tensor,  # (b, seq, seq)
                              ) -> torch.Tensor:  # ()
    mse = mse_loss(output, target, mask)
    masked_output = output * mask.float()
    total_constraint = F.relu(16.0 - masked_output.sum(dim=2)).sum(dim=1).mean()
    return mse + total_constraint


loss_fns: Dict[str, Callable] = {
    'ce': cross_entropy_loss,
    'bce': binary_cross_entropy_with_logits,
    'mrl': margin_ranking_loss,
    'mse': mse_loss,
}
