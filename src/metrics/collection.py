from typing import List

import torch
import torch.nn as nn
import torchmetrics

from datamodule.dataset.base import BaseDataset
from .f1 import MultiPrecision, MultiRecall, MultiF1, SinglePrecision, SingleRecall, SingleF1
from .mse import MeanSquaredError
from .correlation import PearsonCorrcoef, SpearmanCorrcoef
from .auc import AUROC, AveragePrecision


class MetricCollection(nn.Module):
    def __init__(self):
        super().__init__()
        threshold_ratio = 7 / 16
        self.collection = torchmetrics.MetricCollection({
            'single_precision': SinglePrecision(),
            'single_recall': SingleRecall(),
            'single_f1': SingleF1(),
            'multi_precision': MultiPrecision(threshold=threshold_ratio, ignore_null=True),
            'multi_recall': MultiRecall(threshold=threshold_ratio, ignore_null=True),
            'multi_f1': MultiF1(threshold=threshold_ratio, ignore_null=True),
            'multi_f1_with_null': MultiF1(threshold=threshold_ratio, ignore_null=False),
            'mse': MeanSquaredError(),
            'roc_auc': AUROC(threshold=threshold_ratio, ignore_null=True),
            'ap': AveragePrecision(threshold=threshold_ratio, ignore_null=True),
            'pearson': PearsonCorrcoef(),
            'spearman': SpearmanCorrcoef(),
        })

    def update(self,
               example_id: torch.Tensor,
               output: torch.Tensor,
               dataset: BaseDataset,
               ) -> None:
        preds: List[List[float]] = []
        golds: List[List[float]] = []
        assert len(output) == len(example_id)
        for out, eid in zip(output, example_id):
            gold_example = dataset.gold_examples[eid.item()]
            # use gold example for bridging anaphora detection to match both anaphors of gold and prediction
            pred: List[List[float]] = dataset.dump_prediction(out.tolist(), gold_example)
            gold: List[List[float]] = dataset.dump_gold(gold_example)  # -1 for non-candidate
            pred, gold = self._filter_candidates(pred, gold)
            preds += pred
            golds += gold

        for metric in self.collection.values():
            if hasattr(metric, 'max_score'):
                metric.max_score = dataset.max_score

        self.collection.update(preds=preds, target=golds)

    def compute(self):
        return self.collection.compute()

    def reset(self):
        self.collection.reset()

    @staticmethod
    def _filter_candidates(pred: List[List[float]], gold: List[List[float]]):
        assert len(pred) == len(gold)
        pred_, gold_ = [], []
        for ps, gs in zip(pred, gold):
            if not gs:
                assert not ps
                continue
            ps_, gs_ = [], []
            assert len(ps) == len(gs)
            for p, g in zip(ps, gs):
                if g == -1:
                    assert p == -1
                    continue
                ps_.append(p)
                gs_.append(g)
            pred_.append(ps_)
            gold_.append(gs_)
        return pred_, gold_

    @staticmethod
    def _strip_null(ls: List[list]) -> List[list]:
        return list(map(lambda l: l[:-1], ls))

    @staticmethod
    def _gather(tensors: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat(tensors, dim=0)
