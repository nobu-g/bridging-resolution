import torch
import pytest

from model.loss import margin_ranking_loss


def test_margin_ranking_loss():
    output = torch.as_tensor([[[0.1, 0.8, 0.1, 0.4]]])  # (1, 1, 4)
    target = torch.as_tensor([[[0.5, 0.0, 1.0, 0.0]]])  # (1, 1, 4)
    """
    ΔO = 
    [
        [],
        [0.7],
        [  0, -0.7],
        [0.3, -0.4, 0.3],
    ]
    ΔT =
    [
        [],
        [-0.5],
        [ 0.5, 1.0],
        [-0.5,   0, -1.0],
    ]
    -ΔO + ΔT = 
    [
        [],
        [-1.2],
        [ 0.5, 1.7],
        [-0.8, 0.4, -1.3],
    ]
    loss = max(0, sign(ΔT) * (-ΔO + ΔT)) =
    [
        [],
        [1.2],
        [0.5, 1.7],
        [0.8,    0, 1.3],
    ]
    """
    loss = margin_ranking_loss(output, target, torch.ones_like(target))
    assert pytest.approx(loss) == (1.2 + 0.5 + 1.7 + 0.8 + 0 + 1.3) / 6
