import abc
from typing import Tuple, List

import torch
import torch.nn as nn


class SmallModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super(SmallModel, self).__init__()

    @abc.abstractmethod
    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, *args, **kwargs) -> Tuple[torch.Tensor, List[int]]:
        self.eval()
        predict_logits = self.forward(*args, **kwargs).detach().cpu()  # (bs, 2)
        predict_labels = predict_logits.argmax(1).tolist()  # (bs, )

        return predict_logits, predict_labels


class MLSmallModel(SmallModel):
    TASK_ID_MAP = {
        0: 'humor',
        1: 'sarcasm'
    }


class MLPSmallModel2(SmallModel):
    TASK_ID_MAP = {
        0: 'humor',
        1: 'sarcasm'
    }
