import os
from typing import Optional, List, Dict, Any

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, ConcatDataset

from datamodule.dataset.base import BaseDataset


class BridgingDataModule(pl.LightningDataModule):
    def __init__(self,
                 cfg: DictConfig,
                 batch_size: int,
                 num_workers: int = -1,
                 ) -> None:
        super().__init__()
        self.cfg: DictConfig = cfg
        self.batch_size: int = batch_size
        self.num_workers: int = num_workers if num_workers >= 0 else os.cpu_count()
        self.train_dataset = None
        self.valid_datasets: Dict[str, BaseDataset] = {}
        self.test_datasets: Dict[str, BaseDataset] = {}

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        # called on every GPU
        print(f'loading datasets on stage: {stage}')
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING):
            self.train_dataset = ConcatDataset([
                hydra.utils.instantiate(conf, n_jobs=self.num_workers) for conf in self.cfg.train.values()
            ])
        if stage in (TrainerFn.FITTING, TrainerFn.TUNING, TrainerFn.VALIDATING, TrainerFn.TESTING,
                     TrainerFn.PREDICTING):
            self.valid_datasets = {
                corpus: hydra.utils.instantiate(conf, n_jobs=self.num_workers)
                for corpus, conf in self.cfg.valid.items()
            }
        if stage in (TrainerFn.TESTING, TrainerFn.PREDICTING):
            self.test_datasets = {
                corpus: hydra.utils.instantiate(conf, n_jobs=self.num_workers)
                for corpus, conf in self.cfg.test.items()
            }

    def train_dataloader(self) -> DataLoader:
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> List[DataLoader]:
        return [self._get_dataloader(ds, shuffle=False) for ds in self.valid_datasets.values()]

    def test_dataloader(self) -> List[DataLoader]:
        return [self._get_dataloader(ds, shuffle=False) for ds in self.test_datasets.values()]

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        pass

    def _get_dataloader(self, dataset: BaseDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=False,
            collate_fn=self._data_collator
        )

    @staticmethod
    def _data_collator(features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        first: Dict[str, Any] = features[0]
        batch = {}
        for field in first:
            feats = [f[field] for f in features]
            batch[field] = torch.as_tensor(feats)
        return batch

    @property
    def train_data_size(self) -> int:
        return len(self.train_dataset)

    @property
    def valid_data_size(self) -> Dict[str, int]:
        return {c: len(ds) for c, ds in self.valid_datasets.items()}

    @property
    def test_data_size(self) -> Dict[str, int]:
        return {c: len(ds) for c, ds in self.test_datasets.items()}
