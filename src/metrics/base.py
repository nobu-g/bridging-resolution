from typing import List, Optional

import torch
from torchmetrics.metric import Metric


class MetricWrapper(Metric):
    def __init__(self, metric: Metric, dtype: Optional[torch.dtype] = None):
        super().__init__()
        self.metric: Metric = metric
        self.tensor_dtype: Optional[torch.dtype] = dtype

    def update(self,
               preds: List[List[float]],
               target: List[List[float]],
               ) -> None:
        raise NotImplementedError

    def _as_tensor(self, ls: List[float], dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        tensor_dtype = dtype or self.tensor_dtype
        kwargs = {'dtype': tensor_dtype} if tensor_dtype else {}
        t = torch.as_tensor(ls, **kwargs)
        return t.cuda() if torch.cuda.is_available() is True else t

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self):
        return self.metric.reset()
