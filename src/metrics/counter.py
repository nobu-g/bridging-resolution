from typing import List

import torch
import torchmetrics


class Counter(torchmetrics.Metric):
    def __init__(self, ignore_null: bool = True):
        super().__init__()
        self.ignore_null = ignore_null
        self.add_state('counter', default=torch.zeros([], dtype=torch.int), dist_reduce_fx='sum')

    def update(self, preds: List[List[float]], target: List[List[float]]) -> None:
        for ps_, gs_ in zip(preds, target):
            if self.ignore_null is True:
                ps_, gs_ = ps_[:-1], gs_[:-1]
            self.counter += len(ps_)

    def compute(self) -> torch.Tensor:
        return self.counter.float()

    def reset(self):
        self.counter = torch.zeros([], dtype=torch.int)
