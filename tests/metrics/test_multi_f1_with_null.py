from dataclasses import dataclass
from typing import List

import pytest
from metrics.f1 import MultiPrecision, MultiRecall, MultiF1


@dataclass(frozen=True)
class MetricTestCase:
    preds: List[List[List[float]]]
    target: List[List[List[float]]]
    precision: float
    recall: float
    f1: float


case = MetricTestCase(
    preds=[
        [
            [0.8, 0.2, 0.3, 0.9],
            [0.9, 0.9, 0.4, 0.0],
        ],
        [
            [0.3, 0.9, 0.2, 0.5],
            [0.5, 0.9, 0.8, 0.0],
        ],
    ],
    target=[
        [
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        [
            [1, 0, 1, 1],
            [1, 1, 0, 0],
        ],
    ],
    precision=4 / 7,
    recall=4 / 10,
    f1=(2 * 4) / (7 + 10),
)


def test_multi_precision_with_null():
    metric = MultiPrecision(threshold=0.7, ignore_null=False)
    metric.max_score = 1
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.precision)


def test_multi_recall_with_null():
    metric = MultiRecall(threshold=0.7, ignore_null=False)
    metric.max_score = 1
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.recall)


def test_multi_f1_with_null():
    metric = MultiF1(threshold=0.7, ignore_null=False)
    metric.max_score = 1
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.f1)
