from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from ..base import MLSmallModel


class PredictHead(nn.Module):
    def __init__(self, input_dim: int, num_labels: int, dropout: float = 0.1):
        super(PredictHead, self).__init__()

        self.dense = nn.Linear(input_dim, input_dim)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(input_dim, num_labels)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # x.shape = (bs, seq_len, hs)
        if hidden_states.dim() == 3:
            first_token_tensor = hidden_states[:, 0]
        elif hidden_states.dim() == 2:
            first_token_tensor = hidden_states
        else:
            raise ValueError
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        pooled_output = self.dropout(pooled_output)
        # (bs, num_labels)
        logits = self.classifier(pooled_output)
        return logits


class MCWF(MLSmallModel):
    def __init__(self, modals_config: dict, model_config: dict, **kwargs):
        """
        MSWF Multitask Learning
        Args:
            modals_config: 模态配置, {'text': 4096, 'audio': 1024, 'vision': 796 }
            model_config: 模型配置
        """
        super(MCWF, self).__init__()
        assert len(modals_config) >= 1
        self.main_task_id = model_config.get('main_task_id', 0)
        self.main_task_weight = model_config.get('main_task_weight', 1.0)
        # 模态对齐
        aligned_dim = model_config['aligned_dim']
        self.aligned_fc = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(dim, aligned_dim),
                nn.Tanh(),
            )
            for k, dim in modals_config.items()
        })

        self.private_encoders = nn.ModuleDict({
            k: nn.GRU(
                input_size=aligned_dim,
                batch_first=True,
                bidirectional=True,
                hidden_size=aligned_dim // 2,
                dropout=0.1,
                num_layers=2,
            )
            for k, dim in modals_config.items()
        })

        self.task_num = model_config['task_num']

        self.task_tower = nn.ModuleList([
            nn.ModuleDict({
                modal: PredictHead(aligned_dim, model_config['num_labels'])
                for modal in modals_config
            })
            for _ in range(self.task_num)
        ])

        self.task_fc = nn.ModuleList([
            nn.Linear(len(modals_config) * 2, 2)
            for _ in range(self.task_num)
        ])

    def forward(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: Dict[str, torch.Tensor],
            **kwargs
    ):
        # 模态对齐
        aligned_features = {
            modal: layer(input_feature[modal])
            for modal, layer in self.aligned_fc.items()
        }

        # 模态单独处理
        encoder_features = {}
        for modal, gru in self.private_encoders.items():
            final_states = gru(pack_padded_sequence(
                aligned_features[modal],
                input_feature_length[modal].cpu(),
                enforce_sorted=False,
                batch_first=True
            ))[1]
            encoder_features[modal] = torch.cat([final_states[-2], final_states[-1]], dim=-1)

        loss = 0
        record = {
            'humor_loss': 0,
            'sarcasm_loss': 0,
            'humor_label': [],
            'sarcasm_label': [],
            'humor_predict': [],
            'sarcasm_predict': [],
        }
        # 提取不同任务的数据
        for i, (tower, tower_fc) in enumerate(zip(self.task_tower, self.task_fc)):
            modal_logits = []
            for modal, modal_t in encoder_features.items():
                modal_logits.append(tower[modal](modal_t))
            # (bs, 6) -> (bs, 2)
            current_task_predict_logits = tower_fc(torch.cat(modal_logits, dim=-1))
            current_task_predict = current_task_predict_logits.argmax(dim=-1)
            current_task_loss = F.cross_entropy(current_task_predict_logits, label[self.TASK_ID_MAP[i]])
            loss = loss + current_task_loss if i != self.main_task_id else current_task_loss * self.main_task_weight

            record[self.TASK_ID_MAP[i] + '_loss'] = current_task_loss
            record[self.TASK_ID_MAP[i] + '_predict'] = current_task_predict.tolist()
            record[self.TASK_ID_MAP[i] + '_label'] = label[self.TASK_ID_MAP[i]].tolist()
        return {
            'loss': loss,
            **record
        }
