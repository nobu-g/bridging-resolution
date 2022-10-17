import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import hydra
import pytorch_lightning as pl
import torch
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn

from datamodule.datamodule import BridgingDataModule
from datamodule.dataset.base import BaseDataset
from litmodule import BridgingModule
from test import load_config

hf_logging.set_verbosity(hf_logging.ERROR)


# @hydra.main(config_path='../conf', config_name='test')
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '--ckpt', '-c', '--model', '-m', type=str, required=True,
                        help='path to config file saved when training')
    parser.add_argument('overrides', type=str, nargs='*',
                        help='config overrides in a hydra format')
    args = parser.parse_args()

    # Load saved model and config
    model = BridgingModule.load_from_checkpoint(checkpoint_path=args.checkpoint)

    cfg: DictConfig = load_config(model.hparams, config_name='test', overrides=args.overrides)

    # Instantiate lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=False, devices=cfg.devices)

    # Instantiate lightning datamodule
    datamodule = BridgingDataModule(
        cfg=cfg.datamodule,
        num_workers=cfg.num_workers,
        batch_size=cfg.virtual_batch_size // max(1, trainer.num_gpus)
    )
    datamodule.setup(stage=TrainerFn.PREDICTING)

    if cfg.eval_set == 'valid':
        datasets: Dict[str, BaseDataset] = datamodule.valid_datasets
        dataloader = datamodule.val_dataloader()
    elif cfg.eval_set == 'test':
        datasets: Dict[str, BaseDataset] = datamodule.test_datasets
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f'datasets for eval set {cfg.eval_set} not found')

    # Run prediction
    raw_outputs = trainer.predict(model=model, dataloaders=dataloader)
    save_prediction(
        [raw_outputs] if isinstance(raw_outputs[0], torch.Tensor) else raw_outputs,
        datasets,
        Path(cfg.pred_dir)
    )
    save_gold(datasets, Path(cfg.gold_dir))


@pl.utilities.rank_zero_only
def save_prediction(outputs: List[List[Tuple[torch.Tensor, torch.Tensor]]],
                    datasets: Dict[str, BaseDataset],
                    pred_dir: Path,
                    ) -> None:
    assert len(datasets) == len(outputs)
    for (corpus, dataset), outs in zip(datasets.items(), outputs):
        example_id, out = (torch.cat(ts, dim=0).cpu() for ts in zip(*outs))  # (N), (N, seq, seq)
        save_dir = pred_dir / corpus
        save_dir.mkdir(exist_ok=True, parents=True)
        assert len(out) == len(example_id)
        for result, eid in zip(out, example_id):
            example = dataset.gold_examples[eid]
            prediction: List[List[float]] = dataset.dump_prediction(result.tolist(), example)
            save_dir.joinpath(f'{example.doc_id}.json').write_text(
                json.dumps(prediction, sort_keys=False)
            )


@pl.utilities.rank_zero_only
def save_gold(datasets: Dict[str, BaseDataset], gold_dir: Path):
    for corpus, dataset in datasets.items():
        save_dir = gold_dir / corpus
        save_dir.mkdir(exist_ok=True, parents=True)
        for example in dataset.gold_examples:
            gold: List[List[float]] = dataset.dump_gold(example)
            save_dir.joinpath(f'{example.doc_id}.json').write_text(
                json.dumps(gold, sort_keys=False)
            )


if __name__ == '__main__':
    main()
