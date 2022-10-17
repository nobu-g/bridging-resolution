import argparse
from collections import defaultdict
from pathlib import Path
from typing import List, Dict

import hydra
import pytorch_lightning as pl
import torch.cuda
import transformers.utils.logging as hf_logging
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.trainer.states import TrainerFn

from datamodule.datamodule import BridgingDataModule
from litmodule import BridgingModule

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
    datamodule.setup(stage=TrainerFn.TESTING)

    if cfg.eval_set == 'valid':
        dataloader = datamodule.val_dataloader()
    elif cfg.eval_set == 'test':
        dataloader = datamodule.test_dataloader()
    else:
        raise ValueError(f'datasets for eval set {cfg.eval_set} not found')

    # Run test
    raw_results: List[Dict[str, float]] = trainer.test(model=model, dataloaders=dataloader)
    save_results(raw_results, Path(cfg.eval_dir))


def load_config(base_cfg: DictConfig, config_name: str, overrides: List[str]) -> DictConfig:
    cfg = base_cfg
    OmegaConf.set_struct(cfg, False)  # enable to add new key-value pairs

    hydra.initialize(config_path='../conf')
    test_cfg = hydra.compose(config_name=config_name, overrides=overrides)
    cfg.merge_with(test_cfg)
    if overrides:
        for item in overrides:
            key, value = item.split('=')
            if key in ('eval_set', 'devices', 'num_workers'):
                continue
            cfg.pred_dir += f',{key}={value}'
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(',')]
        except ValueError:
            cfg.devices = None
    if torch.cuda.is_available() is False:
        cfg.devices = 0
    return cfg


@pl.utilities.rank_zero_only
def save_results(results: List[Dict[str, float]], save_dir: Path):
    test_results = defaultdict(dict)
    for k, v in [item for result in results for item in result.items()]:
        met, corpus = k.split('/')
        if met in test_results[corpus]:
            assert v == test_results[corpus][met]
        else:
            test_results[corpus][met] = v

    save_dir.mkdir(exist_ok=True, parents=True)
    for corpus, result in test_results.items():
        with save_dir.joinpath(f'{corpus}.csv').open(mode='wt') as f:
            f.write(','.join(result.keys()) + '\n')
            f.write(','.join(f'{v:.6}' for v in result.values()) + '\n')


if __name__ == '__main__':
    main()
