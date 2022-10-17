import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from metrics import MetricCollection


class BridgingModule(pl.LightningModule):
    def __init__(self, hparams: DictConfig):
        super().__init__()
        OmegaConf.resolve(hparams)
        # this line allows accessing init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(hparams)
        self.valid_corpora = list(hparams.datamodule.valid.keys())
        self.test_corpora = list(hparams.datamodule.test.keys())
        self.val_metrics = nn.ModuleDict({corpus: MetricCollection() for corpus in self.valid_corpora + ['all']})
        self.test_metrics = nn.ModuleDict({corpus: MetricCollection() for corpus in self.test_corpora + ['all']})

        self.model: nn.Module = hydra.utils.instantiate(self.hparams.model,
                                                        vocab_size=hparams.model.vocab_size + len(hparams.exophors) + 1)

    def setup(self, stage: Optional[str] = None) -> None:
        pass

    def configure_optimizers(self):
        # Split weights in two groups, one with weight decay and the other not.
        no_decay = ('bias', 'LayerNorm.weight')
        optimizer_grouped_parameters = [
            {
                'params': [
                    p for n, p in self.model.named_parameters() if
                    not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': self.hparams.optimizer.weight_decay,
                'name': 'decay',
            },
            {
                'params': [
                    p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                'weight_decay': 0.0,
                'name': 'no_decay',
            },
        ]
        optimizer = hydra.utils.instantiate(self.hparams.optimizer, params=optimizer_grouped_parameters,
                                            _convert_='partial')
        num_train_examples = sum(
            self.hparams.dataset.num_examples.train[corpus] for corpus in self.hparams.datamodule.train.keys()
        )
        total_steps = math.ceil(num_train_examples / self.hparams.virtual_batch_size) * self.hparams.trainer.max_epochs
        warmup_steps = self.hparams.warmup_steps or total_steps * self.hparams.warmup_ratio
        lr_scheduler = hydra.utils.instantiate(self.hparams.scheduler,
                                               optimizer=optimizer,
                                               num_warmup_steps=warmup_steps,
                                               num_training_steps=total_steps)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'step',
                'frequency': 1,
            }
        }

    # def on_train_start(self) -> None:
    #     experiment = self.trainer.logger.experiment
    #     for logger in experiment if isinstance(experiment, list) else [experiment]:
    #         if logger.__class__ == wandb.sdk.wandb_run.Run:
    #             logger.watch(self.model, log_freq=1)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def training_step(self, batch: dict, batch_idx: int):
        loss, *_ = self(**batch)
        self.log('train_loss', loss, on_step=True, on_epoch=False)
        return loss

    def validation_step(self, batch: dict, batch_idx: int, dataloader_idx: Optional[int] = None):
        corpus = self.valid_corpora[dataloader_idx or 0]
        loss, *output = self(**batch)
        if len(loss.size()) > 0:
            loss = loss.mean()
        output: torch.Tensor = output[0]  # (b, seq, seq)

        met_args = {k: v for k, v in batch.items() if k in ('example_id',)}
        met_args.update({'output': output, 'dataset': self.trainer.val_dataloaders[dataloader_idx or 0].dataset})
        for key in (corpus, 'all'):
            self.val_metrics[key].update(**met_args)
            if key != 'all':
                self.log(f'val_loss/{key}', loss, on_step=False, add_dataloader_idx=False)

    def validation_epoch_end(self, outputs):
        for corpus, metric in self.val_metrics.items():
            for name, value in metric.compute().items():
                self.log(f'val_{name}/{corpus}', value)
            metric.reset()

    @pl.utilities.rank_zero_only
    def on_train_end(self) -> None:
        if not self.trainer.checkpoint_callback.best_model_path:
            return
        save_dir = Path(self.hparams.exp_dir) / self.hparams.run_id
        # save_dir.joinpath('config.yaml').write_text(OmegaConf.to_yaml(self.hparams, resolve=True))
        best_path = save_dir / 'best.ckpt'
        if best_path.exists():
            best_path.unlink()
        actual_best_path = Path(self.trainer.checkpoint_callback.best_model_path)
        assert actual_best_path.parent.resolve() == best_path.parent.resolve()
        best_path.resolve().symlink_to(actual_best_path.name)

    def test_step(self, batch, batch_idx: int, dataloader_idx: Optional[int] = None):
        corpus = self.test_corpora[dataloader_idx or 0]
        loss, *output = self(**batch)
        if len(loss.size()) > 0:
            loss = loss.mean()
        output: torch.Tensor = output[0]  # (b, seq, seq)

        met_args = {k: v for k, v in batch.items() if k in ('example_id',)}
        met_args.update({'output': output, 'dataset': self.trainer.test_dataloaders[dataloader_idx or 0].dataset})
        for key in (corpus, 'all'):
            self.test_metrics[key].update(**met_args)
            if key != 'all':
                self.log(f'test_loss/{key}', loss, on_step=False, add_dataloader_idx=False)

    def test_epoch_end(self, outputs):
        logs = {}
        for corpus, metric in self.test_metrics.items():
            for name, value in metric.compute().items():
                logs[f'{name}/{corpus}'] = value.item()
            metric.reset()
        # set dataloader_idx because metric values are filtered out if dataloader_idx is None
        # https://github.com/PyTorchLightning/pytorch-lightning/blob/b707c677eb06bd00e537a09e62ffa4bf55b4e30a/pytorch_lightning/trainer/connectors/logger_connector/result.py#L539
        self._current_dataloader_idx = 0
        self.log_dict(logs, add_dataloader_idx=False)
        return logs

    def predict_step(self,
                     batch: Dict[str, torch.Tensor],
                     batch_idx: int,
                     dataloader_idx: Optional[int] = None
                     ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, *output = self(**batch)
        example_id: torch.Tensor = self.all_gather(batch['example_id'])  # (process, b) or (b)
        output: torch.Tensor = self.all_gather(output[0])  # (process, b, seq, seq) or (b, seq, seq)
        if len(example_id.size()) > 1:
            assert len(output.size()) == 4
            example_id = example_id.view(-1)
            output = output.view(-1, output.size(2), output.size(3))
        return example_id, output  # (b), (b, seq, seq)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def val_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def test_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        pass
