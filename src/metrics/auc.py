from typing import List

import torch
import torchmetrics

from .base import MetricWrapper


class AUROC(MetricWrapper):
    def __init__(self, threshold: float, ignore_null: bool = True):
        super().__init__(torchmetrics.AUROC(pos_label=1))
        self.threshold_ratio: float = threshold
        self.max_score = None
        self.ignore_null: bool = ignore_null

    def update(self,
               preds: List[List[float]],
               target: List[List[float]],
               ) -> None:
        threshold = self.threshold_ratio * self.max_score
        ps, gs = [], []
        for ps_, gs_ in zip(preds, target):
            if self.ignore_null is True:
                ps_, gs_ = ps_[:-1], gs_[:-1]
            ps += ps_
            gs += [g >= threshold for g in gs_]
        self.metric.update(preds=self._as_tensor(ps, dtype=torch.float), target=self._as_tensor(gs, torch.int8))


class AveragePrecision(MetricWrapper):
    def __init__(self, threshold: float, ignore_null: bool = True):
        super().__init__(torchmetrics.AveragePrecision(pos_label=1))
        self.threshold_ratio: float = threshold
        self.max_score = None
        self.ignore_null: bool = ignore_null

    def update(self,
               preds: List[List[float]],
               target: List[List[float]],
               ) -> None:
        threshold = self.threshold_ratio * self.max_score
        ps, gs = [], []
        for ps_, gs_ in zip(preds, target):
            if self.ignore_null is True:
                ps_, gs_ = ps_[:-1], gs_[:-1]
            ps += ps_
            gs += [g >= threshold for g in gs_]
        self.metric.update(preds=self._as_tensor(ps, dtype=torch.float), target=self._as_tensor(gs, torch.int8))
