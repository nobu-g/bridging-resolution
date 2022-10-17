from pathlib import Path
from typing import Iterable, List, Dict

import hydra
import pytorch_lightning as pl
import transformers.utils.logging as hf_logging
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import LightningLoggerBase

from datamodule.datamodule import BridgingDataModule
from litmodule import BridgingModule
from test import save_results

hf_logging.set_verbosity(hf_logging.ERROR)


@hydra.main(config_path='../conf', config_name='example')
def main(cfg: DictConfig):
    # cfg.devices: '0,1', '0', 0, [0, 1]
    # pl.Trainer accepts: Optional[Union[List[int], str, int]]
    if isinstance(cfg.devices, str):
        try:
            cfg.devices = [int(x) for x in cfg.devices.split(',')]
        except ValueError:
            cfg.devices = None
    cfg.seed = pl.seed_everything(seed=cfg.seed, workers=True)

    # Instantiate lightning module
    model = BridgingModule(hparams=cfg)

    # Instantiate lightning loggers
    loggers: List[LightningLoggerBase] = [
        hydra.utils.instantiate(c, _convert_='all') for c in cfg.get('logger', {}).values()
    ]

    # Instantiate lightning callbacks
    callbacks: List[Callback] = list(map(hydra.utils.instantiate, cfg.get('callbacks', {}).values()))

    # Instantiate lightning trainer
    trainer: pl.Trainer = hydra.utils.instantiate(cfg.trainer, logger=loggers, callbacks=callbacks, devices=cfg.devices)

    # Instantiate lightning datamodule
    datamodule = BridgingDataModule(
        cfg=cfg.datamodule,
        num_workers=cfg.num_workers,
        batch_size=cfg.virtual_batch_size // max(1, trainer.num_gpus)
    )

    # Run training
    trainer.fit(model=model, datamodule=datamodule)

    # Run test
    raw_results: List[Dict[str, float]] = trainer.test(model=model, datamodule=datamodule)
    save_results(raw_results, Path(cfg.run_dir) / f'eval_test')

    wandb.finish()


if __name__ == '__main__':
    main()
