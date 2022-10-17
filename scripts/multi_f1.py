import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn import metrics
from wandb.apis.public import Runs

from util import filter_candidates, list_runs, calc_interval

EXPERIMENTS = ('kwdlc-mrl', 'crowd-mse', 'crowd-kwdlc-mrl', 'crowd-kwdlc-mse')


def get_results(pred_dir: Path, gold_dir: Path, ignore_null: bool = True) -> list[dict[str, list]]:
    pred_files, gold_files = sorted(pred_dir.iterdir()), sorted(gold_dir.iterdir())
    assert len(pred_files) == len(gold_files)
    ret = []
    for pred_file, gold_file in zip(pred_files, gold_files):
        for preds, golds in zip(json.loads(pred_file.read_text()), json.loads(gold_file.read_text())):
            if not preds:
                assert not golds
                continue
            assert len(preds) == len(golds), f'length not match: {len(preds)} vs {len(golds)}'
            if ignore_null is True:
                preds, golds = preds[:-1], golds[:-1]
            ret.append({'votes': preds, 'goldVotes': golds})
    return ret


def calc_multi_f1(pred_dir: Path, gold_dir: Path, threshold: Optional[float]) -> tuple[float, float, float, float]:
    y_score: list[float] = []
    y_true: list[int] = []
    for result in get_results(pred_dir, gold_dir):
        pred_votes, gold_votes = filter_candidates(result['votes'], result['goldVotes'])
        th = 7 / 16
        for pred_vote, gold_vote in zip(pred_votes, gold_votes):
            y_score.append(pred_vote)
            y_true.append(int(gold_vote >= th))

    if threshold is not None:
        y_pred = [y >= threshold for y in y_score]
        return (
            metrics.f1_score(y_true=y_true, y_pred=y_pred).item(),
            threshold,
            metrics.precision_score(y_true=y_true, y_pred=y_pred).item(),
            metrics.recall_score(y_true=y_true, y_pred=y_pred).item(),
        )

    precisions, recalls, thresholds = metrics.precision_recall_curve(y_true, y_score)
    f1_scores = 2 * recalls * precisions / (recalls + precisions + 1e-6)
    max_index = np.argmax(f1_scores)
    return (
        np.max(f1_scores).item(), thresholds[max_index].item(), precisions[max_index].item(), recalls[max_index].item()
    )


def calc_expr(expr: str):
    ret = {}
    exprs: dict[str, Runs] = {exp: list_runs(exp) for exp in EXPERIMENTS}
    for corpus in ('crowd', 'kwdlc'):
        f1s: list[float] = []
        precisions: list[float] = []
        recalls: list[float] = []
        thresholds: list[float] = []
        for run in exprs[expr]:
            run_dir = Path(run.config['run_dir'])
            pred_dir = run_dir / 'pred_valid' / corpus
            gold_dir = run_dir / 'gold_valid' / corpus
            _, threshold, _, _ = calc_multi_f1(pred_dir, gold_dir, threshold=None)
            thresholds.append(threshold)

            pred_dir = run_dir / 'pred_test' / corpus
            gold_dir = run_dir / 'gold_test' / corpus
            f1, _, precision, recall = calc_multi_f1(pred_dir, gold_dir, threshold=threshold)
            f1s.append(f1)
            precisions.append(precision)
            recalls.append(recall)
        f1_mean, f1_interval = calc_interval(pd.Series(f1s))
        precision_mean, precision_interval = calc_interval(pd.Series(precisions))
        recall_mean, recall_interval = calc_interval(pd.Series(recalls))
        print(f'Evaluated on {corpus}: ')
        print(f'  averaged best test Multi-F1: {f1_mean * 100:.03f} ± {f1_interval * 100:.03f}')
        print(f'  averaged best test Multi-P: {precision_mean * 100:.03f} ± {precision_interval * 100:.03f}')
        print(f'  averaged best test Multi-R: {recall_mean * 100:.03f} ± {recall_interval * 100:.03f}')
        print(f'  thresholds: {thresholds}')
        ret[corpus] = {
            'f1': (f1_mean, f1_interval),
            'precision': (precision_mean, precision_interval),
            'recall': (recall_mean, recall_interval),
        }
    return ret


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', default=None, type=str, help='experiment name')
    return parser.parse_args()


def main():
    args = parse_args()
    results = {}
    if args.exp is not None:
        _ = calc_expr(args.exp)
    else:
        for exp in EXPERIMENTS:
            result = calc_expr(exp)
            results[exp] = result
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
