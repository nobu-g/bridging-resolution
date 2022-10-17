import math
import os

import pandas as pd
import wandb
from scipy import stats
from wandb.apis.public import Runs


def filter_candidates(pred: list[float], gold: list[float]):
    pred_, gold_ = [], []
    for p, g in zip(pred, gold):
        if g == -1:
            assert p == -1
            continue
        pred_.append(p)
        gold_.append(g)
    return pred_, gold_


def list_runs(exp: str) -> Runs:
    api = wandb.Api()
    created_at = {'$gte': '2022/1/1'}
    return api.runs(os.environ['WANDB_PROJECT'], filters={'Sweep': None, 'createdAt': created_at, 'group': exp})


def calc_interval(series: pd.Series) -> tuple[float, float]:
    n = len(series)
    mean = series.mean()
    var = series.var()
    if var == 0:
        return mean, 0
    """
    t分布で信頼区間を計算
    alpha: 何パーセント信頼区間か
    df: t分布の自由度
    loc: 平均 X bar
    scale: 標準偏差 s
    """
    lower, upper = stats.t.interval(alpha=0.95,
                                    df=n - 1,
                                    loc=mean,
                                    scale=math.sqrt(var / n))
    return mean, upper - mean
