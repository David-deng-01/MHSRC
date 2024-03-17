# MISA: Modality-Invariant and -Specific Representations for Multimodal Sentiment Analysis
# https://github.com/declare-lab/MISA
from collections import OrderedDict
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils.tools import c_n_m
from ..base import MLSmallModel

ACTIVATE = {
    'relu': nn.ReLU,
    'sigmoid': nn.Sigmoid
}


def _build_aligned_layer(input_size: int, output_size: int, act: str, use_layer_norm: bool = True):
    layers = [nn.Linear(input_size, output_size), ACTIVATE[act]()]
    if use_layer_norm:
        layers.append(nn.LayerNorm(output_size))
    return nn.Sequential(*layers)


def get_mse_loss(pred, real):
    diffs = torch.add(real, -pred)
    n = torch.numel(diffs.data)
    mse = torch.sum(diffs.pow(2)) / n

    return mse


def get_diff_loss(input1, input2):
    batch_size = input1.size(0)
    input1 = input1.view(batch_size, -1)
    input2 = input2.view(batch_size, -1)

    # Zero mean
    input1_mean = torch.mean(input1, dim=0, keepdims=True)
    input2_mean = torch.mean(input2, dim=0, keepdims=True)
    input1 = input1 - input1_mean
    input2 = input2 - input2_mean

    input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
    input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

    input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
    input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

    diff_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

    return diff_loss


def _matchnorm(x1, x2):
    power = torch.pow(x1 - x2, 2)
    summed = torch.sum(power)
    sqrt = summed ** (0.5)
    return sqrt


def _scm(sx1, sx2, k):
    ss1 = torch.mean(torch.pow(sx1, k), 0)
    ss2 = torch.mean(torch.pow(sx2, k), 0)
    return _matchnorm(ss1, ss2)


def get_cmd_loss(x1, x2, n_moments):
    mx1 = torch.mean(x1, 0)
    mx2 = torch.mean(x2, 0)
    sx1 = x1 - mx1
    sx2 = x2 - mx2
    dm = _matchnorm(mx1, mx2)
    scms = dm
    for i in range(n_moments - 1):
        scms += _scm(sx1, sx2, i + 2)
    return scms


class MISA(MLSmallModel):
    def __init__(self, modals_config: dict, model_config: dict, **kwargs):
        """
        Args:
            modals_config: 模态配置
                {'text': 4096, 'audio': 1024, 'visual': 796 }
            model_config: 模型配置
        """
        super(MISA, self).__init__()
        self.modals_config = modals_config
        self.model_config = model_config
        self.main_task_id = model_config.get('main_task_id', 0)
        self.main_task_weight = model_config.get('main_task_weight', 1.0)
        # a, v 的 encoder
        self.rnn1 = nn.ModuleDict({
            modal: nn.LSTM(modal_in, modal_in, bidirectional=True, batch_first=True)
            for modal, modal_in in modals_config.items() if modal != 'text'
        })
        self.rnn2 = nn.ModuleDict({
            modal: nn.LSTM(2 * modal_in, modal_in, bidirectional=True, batch_first=True)
            for modal, modal_in in modals_config.items() if modal != 'text'
        })

        self.ln = nn.ModuleDict({
            modal: nn.LayerNorm(modal_in * 2)
            for modal, modal_in in modals_config.items() if modal != 'text'
        })

        # 不同模态映射到同一维度
        aligned_dim = model_config['aligned_dim']
        self.project = nn.ModuleDict({
            modal: _build_aligned_layer(modal_in if modal == 'text' else modal_in * 4, aligned_dim, 'relu')
            for modal, modal_in in modals_config.items()
        })

        # private encoders
        self.private = nn.ModuleDict({
            modal: _build_aligned_layer(aligned_dim, aligned_dim, 'sigmoid', False)
            for modal in modals_config
        })

        # shared encoder
        self.shared = _build_aligned_layer(aligned_dim, aligned_dim, 'sigmoid', False)

        # reconstruct
        self.recon = nn.ModuleDict({modal: nn.Linear(aligned_dim, aligned_dim) for modal in modals_config})

        # transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=aligned_dim, nhead=2, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 融合
        self.fusion = nn.ModuleList([
            nn.Sequential(
                nn.Linear(aligned_dim * len(modals_config) * 2, aligned_dim * len(modals_config)),
                nn.Dropout(model_config['dropout_rate']), nn.ReLU(),
                nn.Linear(aligned_dim * len(modals_config), model_config['num_classes']),
            )
            for _ in range(model_config['task_num'])
        ])

    def extract_features(self, sequence, lengths, rnn1, rnn2, layer_norm):
        packed_sequence = pack_padded_sequence(sequence, lengths.cpu(), batch_first=True, enforce_sorted=False)

        packed_h1, (final_h1, _) = rnn1(packed_sequence)

        padded_h1, _ = pad_packed_sequence(packed_h1, batch_first=True)
        normed_h1 = layer_norm(padded_h1)
        packed_normed_h1 = pack_padded_sequence(normed_h1, lengths.cpu(), batch_first=True, enforce_sorted=False)

        _, (final_h2, _) = rnn2(packed_normed_h1)
        return final_h1, final_h2

    def forward(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: Dict[str, torch.Tensor],
            **kwargs
    ):
        batch_size = label[self.TASK_ID_MAP[0]].shape[0]

        aligned_feature = OrderedDict()  # utt_t/a/v_orig
        for modal, modal_tensor in input_feature.items():
            modal_length = input_feature_length[modal]
            if modal == 'text':
                # text 模态 在句子长度维度取均值
                max_text_len = modal_tensor.shape[1]
                sent_mask = torch.zeros((batch_size, max_text_len)).to(modal_tensor.device)
                for idx, t_l in enumerate(modal_length):
                    sent_mask[idx, :t_l] = 1

                masked_output = torch.mul(sent_mask.unsqueeze(2), modal_tensor)
                mask_len = torch.sum(sent_mask, dim=1, keepdim=True)
                aligned_feature[modal] = self.project[modal](
                    torch.sum(masked_output, dim=1, keepdim=False) / mask_len)  # (bs, text_in)
            else:
                # extract features from visual modality
                final_h1v, final_h2v = self.extract_features(
                    modal_tensor, modal_length,
                    self.rnn1[modal], self.rnn2[modal], self.ln[modal]
                )
                aligned_feature[modal] = self.project[modal](
                    torch.cat((final_h1v, final_h2v), dim=2).contiguous().view(batch_size, -1))
        # utt_private_t/a/v
        private_feature = OrderedDict()
        for modal, modal_tensor in aligned_feature.items():
            private_feature[modal] = self.private[modal](modal_tensor)

        # utt_shared_t/a/v
        shared_feature = OrderedDict()
        for modal, modal_tensor in aligned_feature.items():
            shared_feature[modal] = self.shared(modal_tensor)

        # For reconstruction
        recon_feature = OrderedDict()
        for modal in self.modals_config:
            recon_feature[modal] = self.recon[modal](private_feature[modal] + shared_feature[modal])

        # 1-LAYER TRANSFORMER FUSION
        # (bs, 6, aligned_dim)
        h = torch.stack(list(private_feature.values()) + list(shared_feature.values()), dim=1)
        # (bs, 6 * aligned_dim)
        h = self.transformer_encoder(h).reshape(batch_size, -1).contiguous()

        diff_loss = 0
        for modal in self.modals_config:
            diff_loss = diff_loss + get_diff_loss(private_feature[modal], shared_feature[modal])
        if len(self.modals_config) >= 2:
            for modal1, modal2 in c_n_m(list(self.modals_config.keys()), 2):
                diff_loss = diff_loss + get_diff_loss(private_feature[modal1], private_feature[modal2])

        recon_loss = 0
        for modal in self.modals_config:
            recon_loss = recon_loss + get_mse_loss(recon_feature[modal], aligned_feature[modal])
        recon_loss = recon_loss / len(self.modals_config)

        cmd_loss = 0
        if len(self.modals_config) >= 2:
            for modal1, modal2 in c_n_m(list(self.modals_config.keys()), 2):
                cmd_loss = cmd_loss + get_cmd_loss(shared_feature[modal1], shared_feature[modal2], 5)
            cmd_loss = cmd_loss / len(c_n_m(list(self.modals_config.keys()), 2))

        cls_loss = 0
        record = {
            'humor_loss': 0,
            'sarcasm_loss': 0,
            'humor_label': [],
            'sarcasm_label': [],
            'humor_predict': [],
            'sarcasm_predict': [],
        }
        for i, classifier in enumerate(self.fusion):
            # (bs, 2)
            current_task_predict_logits = classifier(h)
            current_task_predict = current_task_predict_logits.argmax(dim=-1)
            current_task_loss = F.cross_entropy(current_task_predict_logits, label[self.TASK_ID_MAP[i]])
            cls_loss = cls_loss + current_task_loss if i != self.main_task_id else current_task_loss * self.main_task_weight

            record[self.TASK_ID_MAP[i] + '_loss'] = current_task_loss
            record[self.TASK_ID_MAP[i] + '_predict'] = current_task_predict.tolist()
            record[self.TASK_ID_MAP[i] + '_label'] = label[self.TASK_ID_MAP[i]].tolist()

        total_loss = cls_loss + self.model_config['diff_weight'] * diff_loss + \
                     cmd_loss * self.model_config['sim_weight'] + \
                     recon_loss * self.model_config['recon_weight']

        return {
            'loss': total_loss,
            'cls': cls_loss,
            'diff': diff_loss,
            'cmd': cmd_loss,
            'recon': recon_loss,
            **record
        }
