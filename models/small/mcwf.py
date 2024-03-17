from typing import Dict

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from .base import SmallModel


class MCWF(SmallModel):
    def __init__(self, modals_config: dict, model_config: dict, **kwargs):
        """
        GRU + FC
        Args:
            modals_config: 模态配置, {'text': 4096, 'audio': 1024, 'vision': 796 }
            model_config: 模型配置
        """
        super(MCWF, self).__init__()
        assert len(modals_config) >= 1
        self.model_config = model_config
        self.modals_config = modals_config

        num_class = model_config['num_class']

        # 模态维度对齐
        self.aligned_fc = nn.ModuleDict({
            k: nn.Sequential(
                nn.Linear(dim, model_config['aligned_dim']),
                nn.Tanh(),
            )
            for k, dim in modals_config.items()
        })

        # gru
        self.modal_grus = nn.ModuleDict({
            k: nn.GRU(
                input_size=model_config['aligned_dim'],
                **model_config['gru_config']
            )
            for k, dim in modals_config.items()
        })

        # 模态预测
        classifier_dict = {k: [] for k in modals_config}
        for k in classifier_dict:
            input_dim = model_config['gru_config']['hidden_size'] * 2 \
                if model_config['gru_config']['bidirectional'] \
                else model_config['gru_config']['hidden_size']
            classifier_dict[k].extend([
                nn.Linear(input_dim, input_dim),
                nn.Dropout(0.1), nn.ReLU(),
                nn.Linear(input_dim, num_class)]
            )
        self.classifier = nn.ModuleDict({
            k: nn.Sequential(*classifier_dict[k])
            for k in classifier_dict
        })

        # 融合预测
        concat_dim = len(modals_config) * num_class
        self.fc = nn.Linear(concat_dim, num_class)

    def forward(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: torch.Tensor,
            **kwargs
    ):
        # 模态对齐
        aligned_features = {
            modal: layer(input_feature[modal])
            for modal, layer in self.aligned_fc.items()
        }

        # 模态单独处理
        encoder_features = {}
        for modal, gru in self.modal_grus.items():
            final_states = gru(pack_padded_sequence(
                aligned_features[modal],
                input_feature_length[modal].cpu(),
                enforce_sorted=False,
                batch_first=True
            ))[1]
            encoder_features[modal] = torch.cat([final_states[-2], final_states[-1]], dim=-1)

        # 模态过classifier
        predict_output = [
            self.classifier[k](v)  # (bs, 2)
            for k, v in encoder_features.items()  # v.shape = (bs, dim)
        ]
        # (bs, 2)
        logits = self.fc(torch.cat(predict_output, dim=-1))
        if self.model_config['num_class'] == 2:
            loss = torch.nn.functional.cross_entropy(logits, label)
        else:
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, label)
        return {'loss': loss, 'logits': logits}
