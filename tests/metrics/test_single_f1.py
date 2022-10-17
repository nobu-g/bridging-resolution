from dataclasses import dataclass
from typing import List

import pytest
from metrics.f1 import SinglePrecision, SingleRecall, SingleF1


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
            [0.82, 0.25, 0.34, 0.95],  # 3 (0)
            [0.96, 0.99, 0.43, 0.04],  # 1
        ],
        [
            [0.32, 0.27, 0.97, 0.53],  # 2
            [0.57, 0.90, 0.84, 0.03],  # 2
        ],
    ],
    target=[
        [
            [0.06, 0.51, 0.13, 0.90],  # 3 (1)
            [0.88, 0.43, 0.58, 0.95],  # 3 (0)
        ],
        [
            [0.65, 0.01, 0.88, 0.25],  # 2
            [0.82, 0.20, 0.36, 0.84],  # 3 (0)
        ],
    ],
    precision=1 / 3,
    recall=1 / 1,
    f1=(2 * 1) / (3 + 1),
)


def test_single_precision():
    metric = SinglePrecision()
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.precision)


def test_single_recall():
    metric = SingleRecall()
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.recall)


def test_single_f1():
    metric = SingleF1()
    for preds, target in zip(case.preds, case.target):
        metric.update(
            preds=preds,
            target=target,
        )
    assert metric.compute().item() == pytest.approx(case.f1)
