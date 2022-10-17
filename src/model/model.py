from typing import Tuple, Callable

import torch
import torch.nn as nn
from transformers import AutoModel, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from model.loss import loss_fns
from datamodule.example.base import LearningMethod


class BaselineModel(nn.Module):
    """Naive Baseline Model"""

    def __init__(self,
                 bert_model: str,
                 vocab_size: int,
                 dropout: float,
                 method: str,
                 loss_fn: str,
                 **_
                 ) -> None:
        super().__init__()
        self.method = LearningMethod(method)
        self.loss_fn: Callable = loss_fns[loss_fn]

        self.bert: PreTrainedModel = AutoModel.from_pretrained(bert_model)
        self.bert.resize_token_embeddings(vocab_size)
        self.dropout = nn.Dropout(dropout)

        mid_hidden_size = 1024

        self.l_ana = nn.Linear(self.bert.config.hidden_size, mid_hidden_size)
        self.l_ant = nn.Linear(self.bert.config.hidden_size, mid_hidden_size)
        self.out = nn.Linear(mid_hidden_size, 1, bias=True)

    def forward(self,
                input_ids: torch.Tensor,       # (b, seq)
                attention_mask: torch.Tensor,  # (b, seq)
                segment_ids: torch.Tensor,     # (b, seq)
                mask: torch.Tensor,            # (b, seq, seq)
                target: torch.Tensor,          # (b, seq, seq)
                **_
                ) -> Tuple[torch.Tensor, torch.Tensor]:  # (), (b, seq, seq)
        # batch_size, sequence_len = input_ids.size()
        # (b, seq, hid)
        bert_out: BaseModelOutputWithPoolingAndCrossAttentions
        bert_out = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)

        h_ana = self.l_ana(self.dropout(bert_out.last_hidden_state))  # (b, seq, hid)
        h_ant = self.l_ant(self.dropout(bert_out.last_hidden_state))  # (b, seq, hid)
        h = torch.tanh(self.dropout(h_ana.unsqueeze(2) + h_ant.unsqueeze(1)))  # (b, seq, seq, hid)
        # -> (b, seq, seq, 1) -> (b, seq, seq)
        output = self.out(h).squeeze(-1)

        loss = self.loss_fn(output, target, mask)  # ()

        return loss, output
