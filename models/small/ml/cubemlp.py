# CubeMLP: An MLP-based Model for Multimodal Sentiment Analysis and Depression Estimation
# https://github.com/kiva12138/CubeMLP
from typing import List, Dict

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from ..base import MLSmallModel


def _get_activation_function(activation):
    activation_dict = {
        "elu": F.elu,
        "gelu": F.gelu,
        "hardshrink": F.hardshrink,
        "hardtanh": F.hardtanh,
        "leakyrelu": F.leaky_relu,
        "prelu": F.prelu,
        "relu": F.relu,
        "rrelu": F.rrelu,
        "tanh": F.tanh,
    }
    return activation_dict[activation]


def _get_output_dim(features_compose_t, features_compose_k, d_out, t_out, k_out):
    if features_compose_t in ['mean', 'sum']:
        classify_dim = d_out
    elif features_compose_t == 'cat':
        classify_dim = d_out * t_out
    else:
        raise NotImplementedError

    if features_compose_k in ['mean', 'sum']:
        classify_dim = classify_dim
    elif features_compose_k == 'cat':
        classify_dim = classify_dim * k_out
    else:
        raise NotImplementedError
    return classify_dim


class MLP(nn.Module):
    def __init__(self, activate, d_in, d_hidden, d_out, bias):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=bias)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=bias)
        self.activation = _get_activation_function(activate)

    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class MLPsBlock(nn.Module):
    def __init__(self, activate, d_ins, d_hiddens, d_outs, dropouts, bias, ln_first=False, res_project=False):
        super(MLPsBlock, self).__init__()
        self.mlp_l = MLP(activate, d_ins[0], d_hiddens[0], d_outs[0], bias)
        self.mlp_k = MLP(activate, d_ins[1], d_hiddens[1], d_outs[1], bias)
        self.mlp_d = MLP(activate, d_ins[2], d_hiddens[2], d_outs[2], bias)
        self.dropout_l = nn.Dropout(p=dropouts[0])
        self.dropout_k = nn.Dropout(p=dropouts[1])
        self.dropout_d = nn.Dropout(p=dropouts[2])
        if ln_first:
            self.ln_l = nn.LayerNorm(d_ins[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_ins[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_ins[2], eps=1e-6)
        else:
            self.ln_l = nn.LayerNorm(d_outs[0], eps=1e-6)
            self.ln_k = nn.LayerNorm(d_outs[1], eps=1e-6)
            self.ln_d = nn.LayerNorm(d_outs[2], eps=1e-6)

        self.ln_fist = ln_first
        self.res_project = res_project
        if not res_project:
            assert d_ins[0] == d_outs[
                0], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[1] == d_outs[
                1], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
            assert d_ins[2] == d_outs[
                2], "Error from MLPsBlock: If using projection for residual, d_in should be equal to d_out."
        else:
            self.res_projection_l = nn.Linear(d_ins[0], d_outs[0], bias=False)
            self.res_projection_k = nn.Linear(d_ins[1], d_outs[1], bias=False)
            self.res_projection_d = nn.Linear(d_ins[2], d_outs[2], bias=False)

    # x: [bs, l, k, d] k=modalityKinds mask: [bs, l]
    def forward(self, x, mask=None):
        if mask is not None:
            print("Warning from MLPsBlock: If using mask, d_in should be equal to d_out.")
        if self.ln_fist:
            x = self.forward_ln_first(x, mask)
        else:
            x = self.forward_ln_last(x, mask)
        return x

    def forward_ln_first(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.ln_l(x.permute(0, 2, 3, 1))
        x = self.mlp_l(x, None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0)  # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l

        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.ln_k(x.permute(0, 1, 3, 2))
        x = self.dropout_k(self.mlp_k(x, None).permute(0, 1, 3, 2))
        x = x + residual_k

        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.ln_d(x)
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d

        return x

    def forward_ln_last(self, x, mask):
        if self.res_project:
            residual_l = self.res_projection_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        else:
            residual_l = x
        x = self.mlp_l(x.permute(0, 2, 3, 1), None).permute(0, 3, 1, 2)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1).unsqueeze(-1).bool(), 0.0)  # Fill mask=True to 0.0
        x = self.dropout_l(x)
        x = x + residual_l
        x = self.ln_l(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        if self.res_project:
            residual_k = self.res_projection_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
        else:
            residual_k = x
        x = self.dropout_k(self.mlp_k(x.permute(0, 1, 3, 2), None).permute(0, 1, 3, 2))
        x = x + residual_k
        x = self.ln_k(x.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)

        if self.res_project:
            residual_d = self.res_projection_d(x)
        else:
            residual_d = x
        x = self.dropout_d(self.mlp_d(x, None))
        x = x + residual_d
        x = self.ln_d(x)

        return x


class MLPEncoder(nn.Module):
    def __init__(
            self,
            activate: str,
            d_in: List[int],  # [length, modal_num, aligned_dim]
            d_hiddens: List[List[int]],
            d_outs: List[List[float]],
            dropouts: List[float],
            bias: bool,
            res_project: List[bool],
            ln_first=False,
    ):
        super(MLPEncoder, self).__init__()
        assert len(d_hiddens) == len(d_outs) == len(res_project)
        self.layers_stack = nn.ModuleList([
            MLPsBlock(
                activate=activate,
                d_ins=d_in if i == 0 else d_outs[i - 1],
                d_hiddens=d_hiddens[i],
                d_outs=d_outs[i],
                dropouts=dropouts,
                bias=bias,
                ln_first=ln_first,
                res_project=res_project[i]
            )
            for i in range(len(d_hiddens))
        ])

    def forward(self, x, mask=None):
        for enc_layer in self.layers_stack:
            x = enc_layer(x, mask)
        return x


class CubeMLP(MLSmallModel):

    def __init__(self, modals_config: dict, model_config: dict):
        """
        Args:
            modals_config: 模态配置
                {'text': 4096, 'audio': 1024, 'visual': 796 }
            model_config: 模型配置
        """
        super().__init__()
        self.main_task_id = model_config.get('main_task_id', 0)
        self.main_task_weight = model_config.get('main_task_weight', 1.0)
        self.modals_config = modals_config
        self.model_config = model_config

        # 不同模态映射到同一个特征维度
        aligned_dim = model_config['aligned_dim']
        if 'text' in modals_config:
            self.text_align_fc = nn.Linear(modals_config['text'], aligned_dim, bias=False)

        self.rnn = nn.ModuleDict({
            modal: nn.GRU(modal_in, aligned_dim, 2, bidirectional=True, batch_first=True)
            for modal, modal_in in modals_config.items()
            if modal != 'text'
        })

        self.ln = nn.ModuleDict({
            modal: nn.LayerNorm(aligned_dim, eps=1e-6)
            for modal, modal_in in modals_config.items()
            if modal != 'text'
        })

        self.dropout = nn.ModuleDict({
            modal: nn.Dropout(model_config['dropouts'][i])
            for i, modal in enumerate(modals_config)
        })

        self.mlp_encoder = MLPEncoder(
            activate=model_config['activate'],
            d_in=[model_config['time_len'], 3, aligned_dim],
            d_hiddens=model_config['d_hiddens'],
            d_outs=model_config['d_outs'],
            dropouts=model_config['dropout_mlp'],
            bias=model_config['bias'],
            ln_first=model_config['ln_first'],
            res_project=model_config['res_project']
        )
        classify_dim = _get_output_dim(
            model_config['features_compose_t'],
            model_config['features_compose_k'],
            model_config['d_outs'][-1][2],
            model_config['d_outs'][-1][0],
            model_config['d_outs'][-1][1]
        )

        task_num = model_config['task_num']
        self.classifier = nn.ModuleList([
            nn.Linear(classify_dim, model_config['num_class'])
            if classify_dim <= 128 else nn.Sequential(
                nn.Linear(classify_dim, 128),
                nn.ReLU(),
                nn.Dropout(model_config['dropouts'][3]),
                nn.Linear(128, model_config['num_class']),
            )
            for _ in range(task_num)
        ])

    def forward(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: Dict[str, torch.Tensor],
            **kwargs
    ):
        """
        Args:
            input_feature:
                {text: (bs, seq_len, text_in), audio: (bs, audio_len, audio_in), visual: (bs, visual_len, visual_in)}
            input_feature_length: {text: (bs, ), audio: (bs, ), visual: (bs)}
            label: {'humor': (bs, ), 'sarcasm': (bs, )}
        """
        batch_size = label[self.TASK_ID_MAP[0]].shape[0]

        # 数据预处理
        processed_feature = {}
        for modal, modal_tensor in input_feature.items():
            if modal == 'text':
                processed_feature[modal] = modal_tensor[:, :self.model_config['time_len'], :]
            else:
                _modal_arr = []
                for i in range(batch_size):
                    modal_l = input_feature_length[modal][i]
                    t_l = input_feature_length['text'][i] \
                        if 'text' in input_feature \
                        else self.model_config['time_len']
                    _modal_arr.append(modal_tensor[i, :modal_l].mean(0, keepdim=True).repeat(t_l, 1))
                processed_feature[modal] = pad_sequence(_modal_arr, padding_value=0, batch_first=True)

        # 模态对齐
        aligned_feature = {}
        for modal, modal_tensor in processed_feature.items():
            if modal == 'text':
                aligned_feature[modal] = self.dropout[modal](self.text_align_fc(modal_tensor))
            else:
                if 'text' not in processed_feature:
                    _length = torch.LongTensor([self.model_config['time_len']] * batch_size)
                    total_length = self.model_config['time_len']
                else:
                    _length = input_feature_length['text'].cpu()
                    total_length = input_feature['text'].shape[1]

                _tensor = pack_padded_sequence(modal_tensor, _length, batch_first=True, enforce_sorted=False)
                self.rnn[modal].flatten_parameters()
                packed_modal_tensor = self.rnn[modal](_tensor)[0]
                # (bs, seq_len/time_len, 2 * aligned_dim)
                t = pad_packed_sequence(
                    packed_modal_tensor,
                    batch_first=True,
                    total_length=total_length
                )[0]
                # (bs, seq_len/time_len, aligned_dim)
                t = torch.stack(torch.split(t, self.model_config['aligned_dim'], dim=-1), -1).sum(-1)
                aligned_feature[modal] = self.dropout[modal](F.relu(self.ln[modal](t)))

        # 所有模态数据充填到time_len
        padding_feature = {}
        for modal, modal_tensor in aligned_feature.items():
            padding_length = self.model_config['time_len'] - modal_tensor.shape[1]
            padding_feature[modal] = F.pad(modal_tensor, (0, 0, 0, padding_length, 0, 0), "constant", 0)

        # (bs, time_len, 3, aligned_dim)
        x = torch.stack(list(padding_feature.values()), dim=2)
        # (bs, 10, 3, 32)
        x = self.mlp_encoder(x, mask=None)

        # (bs, 3, aligned_dim)
        fused_features = x.mean(dim=1)
        # (bs, 3 * aligned_dim)
        fused_features = torch.cat(torch.split(fused_features, 1, dim=1), dim=-1).squeeze(1)

        # Predictions
        loss = 0
        record = {
            'humor_loss': 0,
            'sarcasm_loss': 0,
            'humor_label': [],
            'sarcasm_label': [],
            'humor_predict': [],
            'sarcasm_predict': [],
        }

        for i, classifier in enumerate(self.classifier):
            # (bs, 2)
            current_task_predict_logits = classifier(fused_features)
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
