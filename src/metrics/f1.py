from typing import List

import numpy as np
import torch
from torchmetrics import F1, Precision, Recall, Metric

from .base import MetricWrapper


class ThresholdMetric(MetricWrapper):
    def __init__(self, metric: Metric, threshold: float, ignore_null: bool = True):
        super().__init__(metric, dtype=torch.bool)
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
            ps += [p >= threshold for p in ps_]
            gs += [g >= threshold for g in gs_]
        self.metric.update(preds=self._as_tensor(ps), target=self._as_tensor(gs))


class MultiPrecision(ThresholdMetric):
    def __init__(self, threshold: float, ignore_null: bool = True):
        super().__init__(Precision(average='micro', multiclass=False), threshold, ignore_null=ignore_null)


class MultiRecall(ThresholdMetric):
    def __init__(self, threshold: float, ignore_null: bool = True):
        super().__init__(Recall(average='micro', multiclass=False), threshold, ignore_null=ignore_null)


class MultiF1(ThresholdMetric):
    def __init__(self, threshold: float, ignore_null: bool = True):
        super().__init__(F1(average='micro', multiclass=False), threshold, ignore_null=ignore_null)


class ArgmaxMetric(MetricWrapper):
    BP_LEVEL_NULL_IDX = 1024

    def __init__(self, metric: Metric):
        super().__init__(metric, dtype=torch.int16)

    def update(self,
               preds: List[List[float]],
               target: List[List[float]],
               ) -> None:
        ps, gs = [], []
        for ps_, gs_ in zip(preds, target):
            assert len(ps_) == len(gs_)
            null_idx = len(gs_) - 1  # the index of the last element
            pred_idx: int = np.array(ps_).argmax().item()
            gold_idxs = np.argwhere(gs_ == np.amax(gs_)).flatten().tolist()
            gold_idx = pred_idx if pred_idx in gold_idxs else gold_idxs[0]
            ps.append(pred_idx if pred_idx != null_idx else self.BP_LEVEL_NULL_IDX)  # ignore null
            gs.append(gold_idx if gold_idx != null_idx else self.BP_LEVEL_NULL_IDX)  # ignore null
        # avoid "ValueError: The `ignore_index` 1024 is not valid for inputs with 39 classes"
        if self.BP_LEVEL_NULL_IDX not in ps and self.BP_LEVEL_NULL_IDX not in gs:
            ps.append(self.BP_LEVEL_NULL_IDX)
            gs.append(self.BP_LEVEL_NULL_IDX)

        self.metric.update(preds=self._as_tensor(ps), target=self._as_tensor(gs))


class SinglePrecision(ArgmaxMetric):
    def __init__(self):
        super().__init__(Precision(average='micro', ignore_index=self.BP_LEVEL_NULL_IDX))


class SingleRecall(ArgmaxMetric):
    def __init__(self):
        super().__init__(Recall(average='micro', ignore_index=self.BP_LEVEL_NULL_IDX))


class SingleF1(ArgmaxMetric):
    def __init__(self):
        super().__init__(F1(average='micro', ignore_index=self.BP_LEVEL_NULL_IDX))
