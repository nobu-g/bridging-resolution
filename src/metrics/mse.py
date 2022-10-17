from typing import List

import torch
import torchmetrics

from .base import MetricWrapper


class MeanSquaredError(MetricWrapper):
    def __init__(self, ignore_null: bool = False):
        super().__init__(torchmetrics.MeanSquaredError(), dtype=torch.float)
        self.ignore_null = ignore_null

    def update(self,
               preds: List[List[float]],
               target: List[List[float]],
               ) -> None:
        ps, gs = [], []
        for ps_, gs_ in zip(preds, target):
            if self.ignore_null is True:
                ps_, gs_ = ps_[:-1], gs_[:-1]
            ps += ps_
            gs += gs_
        self.metric.update(preds=self._as_tensor(ps), target=self._as_tensor(gs))
