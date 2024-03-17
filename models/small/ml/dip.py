# DIP: Dual Incongruity Perceiving Network for Sarcasm Detection
# https://github.com/downdric/MSD
import math
from queue import Queue
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base import MLSmallModel


def _build_align_layer(_in: int, _out: int):
    return nn.Sequential(
        nn.Linear(_in, _out),
        nn.Tanh()
    )


def _build_correlation_conv():
    return nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=1, padding=1),
        nn.Conv2d(64, 1, 3, stride=1, padding=1),
        nn.ReLU()
    )


def get_mean_by_length(x: torch.Tensor, length: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 3
    assert x.shape[0] == length.shape[0]
    arr = []
    for _x, _l in zip(x, length):
        arr.append(_x[:_l].mean(dim=0, keepdim=True))
    mean_x = torch.cat(arr, dim=0)
    return mean_x


def fusion(embeddings1, embeddings2, strategy):
    assert strategy in ['sum', 'product', 'concat']
    if strategy == 'sum':
        return (embeddings1 + embeddings2) / 2
    elif strategy == 'product':
        return embeddings1 * embeddings2
    else:
        return torch.cat([embeddings1, embeddings2], dim=1)


class DIP(MLSmallModel):
    def __init__(self, modals_config: dict, model_config: dict, *args, **kwargs):
        """
        Args:
            modals_config: 模态配置
                {'text': 4096, 'audio': 1024, 'visual': 796 }
            model_config: 模型配置
        """
        super(DIP, self).__init__()
        assert 'text' in modals_config
        self.main_task_id = model_config.get('main_task_id', 0)
        self.main_task_weight = model_config.get('main_task_weight', 1.0)
        self.model_config = model_config
        self.modals_config = modals_config

        # 模态对齐
        aligned_dim = model_config['aligned_dim']
        self.aligned_feature_module = nn.ModuleDict({
            modal: _build_align_layer(modal_in, aligned_dim)
            for modal, modal_in in modals_config.items()
        })

        # 不同模态之间的注意力提取
        if len(modals_config) >= 2:
            self.combine_group = list(zip(
                list(modals_config.keys()),  # [text, audio, visual]
                list(modals_config.keys())[1:] + list(modals_config.keys())[:1]  # [audio, visual, text]
            ))
            self.modal_inter_correlation_conv = nn.ModuleDict({
                f'{modal1}_{modal2}_correlation_conv': _build_correlation_conv()
                for modal1, modal2 in self.combine_group
            })

        # 基于文本预测情感极性
        self.task_num = task_num = model_config['task_num']
        self.sentiment_fc1 = nn.ModuleList([
            nn.Linear(aligned_dim, aligned_dim, bias=True)
            for _ in range(task_num)
        ])
        self.sentiment_fc2 = nn.ModuleList([
            nn.Linear(aligned_dim, 1, bias=True)
            for _ in range(task_num)
        ])

        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.ReLU()

        self.memory_length = model_config['memory_length']

        if len(modals_config) >= 2:
            self.modal_inter_sarcasm_bank = [
                {
                    f'{modal1}_{modal2}_sarcasm_bank': Queue(maxsize=self.memory_length)
                    for modal1, modal2 in self.combine_group
                }
                for _ in range(task_num)
            ]
            self.modal_inter_non_sarcasm_bank = [
                {
                    f'{modal1}_{modal2}_non_sarcasm_bank': Queue(maxsize=self.memory_length)
                    for modal1, modal2 in self.combine_group
                }
                for _ in range(task_num)
            ]

        self.multimodal_fusion = model_config["multimodal_fusion"]
        self.multilevel_fusion = model_config["multilevel_fusion"]
        if len(modals_config) >= 2:
            if self.multilevel_fusion != 'concat' and self.multimodal_fusion != 'concat':
                in_dim = len(self.combine_group) * aligned_dim
            elif self.multilevel_fusion == 'concat' and self.multimodal_fusion == 'concat':
                in_dim = len(self.combine_group) * 4 * aligned_dim
            else:
                in_dim = len(self.combine_group) * 2 * aligned_dim
        else:
            in_dim = aligned_dim

        self.final_fc = nn.ModuleList([nn.Linear(in_dim, 1, bias=True) for _ in range(task_num)])

    def _modal_inter_attention(
            self,
            modal1: torch.Tensor,
            modal2: torch.Tensor,
            modal1_name: str,
            modal2_name: str
    ):
        # (bs, modal1_len, modal2_len)
        modal1_modal2_attention_map = torch.bmm(modal1, modal2.transpose(1, 2))
        # (bs, modal1_len, modal2_len)
        module = self.modal_inter_correlation_conv[f'{modal1_name}_{modal2_name}_correlation_conv']
        modal1_modal2_attention_map = module(modal1_modal2_attention_map.unsqueeze(1)).squeeze()

        modal1_modal2_modal1_attn = modal1_modal2_attention_map.mean(2).sigmoid()  # (bs, modal1_len)
        modal1_modal2_modal2_attn = modal1_modal2_attention_map.mean(1).sigmoid()  # (bs, modal2_len)

        return {
            f'{modal1_name}_{modal2_name}_{modal1_name}_attn': modal1_modal2_modal1_attn,
            f'{modal1_name}_{modal2_name}_{modal2_name}_attn': modal1_modal2_modal2_attn,
        }

    def do_aligned_features(self, input_feature: Dict[str, torch.Tensor]):
        # 模态对齐  (bs, seq/audio/visual_len, aligned_dim)
        aligned_features = {
            modal: layer(input_feature[modal])
            for modal, layer in self.aligned_feature_module.items()
        }
        return aligned_features

    def get_sentiment_loss(self, text_sentiment: torch.Tensor, attn_features, input_feature_length, i):
        # sentiment model
        text_sentiment_loss = 0
        for idx in range(text_sentiment.shape[0]):
            # (seq_len, )
            cur_text_sentiment = text_sentiment[idx, :input_feature_length['text'][idx]]
            # (seq_len, aligned_dim)
            cur_text_embeddings = attn_features['text'][idx, :input_feature_length['text'][idx]]

            # (seq_len, aligned_dim)
            predicted_text_sentiment = self.sentiment_fc1[i](cur_text_embeddings)
            predicted_text_sentiment = self.dropout(self.activation(predicted_text_sentiment))
            # (seq_len, )
            predicted_text_sentiment = self.sentiment_fc2[i](predicted_text_sentiment).squeeze(1)
            mask = torch.ones_like(cur_text_sentiment)
            mask[cur_text_sentiment == 0] = 0
            predicted_text_sentiment = predicted_text_sentiment * mask

            text_sentiment_loss = text_sentiment_loss + F.mse_loss(
                predicted_text_sentiment, cur_text_sentiment
            )

        text_sentiment_loss = text_sentiment_loss / text_sentiment.shape[0]
        return text_sentiment_loss

    def _process_more_than_one_modal(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: Dict[str, torch.Tensor],
            text_sentiment: torch.Tensor,
            **kwargs
    ):
        batch_size = label[self.TASK_ID_MAP[0]].shape[0]

        # 模态对齐  (bs, seq/audio/visual_len, aligned_dim)
        aligned_features = self.do_aligned_features(input_feature)

        # 6个key
        # text_audio_text_attn, text_audio_audio_attn
        # visual_text_text_attn, visual_text_visual_attn
        # audio_visual_audio_attn, audio_visual_visual_attn
        modal_inter_attn_dict = {}
        for modal1, modal2 in self.combine_group:
            modal_inter_attn_dict.update(
                self._modal_inter_attention(
                    aligned_features[modal1], aligned_features[modal2],
                    modal1, modal2
                )
            )

        attn_features = {}  # key 是 {modal}, value = (bs, modal_len, aligned_dim)
        for modal, modal_tensor in aligned_features.items():
            right_keys = list(filter(lambda _k: _k.count(modal) == 2, modal_inter_attn_dict.keys()))
            # (bs, 2, modal_len) -> (bs, modal_len, 1)
            right_attn_value = torch.stack([
                modal_inter_attn_dict[_k] for _k in right_keys
            ], dim=1).mean(1).unsqueeze(-1)
            # (bs, modal_len, aligned_dim)
            attn_features[modal] = modal_tensor * right_attn_value

        # key 是 {modal}, value = (bs, aligned_dim)
        modal_cls_features = {
            modal: get_mean_by_length(modal_tensor, input_feature_length[modal])
            for modal, modal_tensor in attn_features.items()
        }

        # semantic model, value = (bs, aligned_dim)
        semantic_modal_embeddings = {}
        for modal, modal_cls in modal_cls_features.items():
            # (aligned_dim, )
            variance = F.normalize(torch.var(modal_cls, dim=0), dim=-1)
            semantic_modal_embeddings[modal] = modal_cls + modal_cls * variance.unsqueeze(0).repeat(batch_size, 1)

        # key = f'{modal1}_{modal2}_sims', value = (bs, )
        modal_inter_sims = {
            f'{modal1}_{modal2}_sims': F.cosine_similarity(
                semantic_modal_embeddings[modal1],
                semantic_modal_embeddings[modal2],
                eps=1e-6, dim=1
            )
            for modal1, modal2 in self.combine_group
        }

        #########################################################################################
        # sentiment model
        text_sentiment_loss = sum([
            self.get_sentiment_loss(text_sentiment, attn_features, input_feature_length, i)
            for i in range(self.task_num)
        ]) / 2.0

        # key 是 {modal}, value = (bs, aligned_dim)
        modal_cls_sentiment_embedding = [
            {
                modal: self.sentiment_fc1[i](modal_tensor)
                for modal, modal_tensor in modal_cls_features.items()
            }
            for i in range(self.task_num)
        ]

        with torch.no_grad():
            # key 是 {modal}, value = (bs, 1)
            modal_cls_sentiment = [
                {
                    modal: self.sentiment_fc2[i](self.activation(self.dropout(self.activation(modal_tensor))))
                    for modal, modal_tensor in modal_cls_sentiment_embedding[i].items()
                }
                for i in range(self.task_num)
            ]
            # key 是 f'{modal1}_{modal2}_contrast_label', value = (bs, bs)
            modal_inter_contrast_label = [{} for _ in range(self.task_num)]
            for i in range(self.task_num):
                for modal1, modal2 in self.combine_group:
                    # (bs, bs)
                    modal1_modal2_contrast_label = torch.abs(
                        modal_cls_sentiment[i][modal1] - modal_cls_sentiment[i][modal2])
                    modal1_modal2_contrast_label = torch.exp(-modal1_modal2_contrast_label)
                    modal1_modal2_contrast_label = modal1_modal2_contrast_label / modal1_modal2_contrast_label.sum(1,
                                                                                                                   keepdim=True)
                    modal_inter_contrast_label[i][f'{modal1}_{modal2}_contrast_label'] = modal1_modal2_contrast_label

        # key 是 f'{modal1}_{modal2}_sim', value = (bs, bs)
        modal_inter_sim = [{} for _ in range(self.task_num)]
        for i in range(self.task_num):
            for modal1, modal2 in self.combine_group:
                modal1_modal2_sim = torch.exp(torch.mm(
                    F.normalize(modal_cls_sentiment_embedding[i][modal1], dim=1),
                    F.normalize(modal_cls_sentiment_embedding[i][modal2], dim=1).t()
                ) / 0.2)
                modal1_modal2_sim = modal1_modal2_sim / modal1_modal2_sim.sum(1, keepdim=True)
                modal_inter_sim[i][f'{modal1}_{modal2}_sim'] = modal1_modal2_sim

        # scalar
        sentiment_contrast_loss = 0
        for i in range(self.task_num):
            for modal1, modal2 in self.combine_group:
                sentiment_contrast_loss = sentiment_contrast_loss + F.kl_div(
                    modal_inter_sim[i][f'{modal1}_{modal2}_sim'].log(),
                    modal_inter_contrast_label[i][f'{modal1}_{modal2}_contrast_label'],
                    reduction='batchmean'
                )

        # key 是 f'{modal1}_{modal2}_lambda_sentiment', value = scalar
        modal_inter_lamda_sentiment = [
            {
                f'{modal1}_{modal2}_lambda_sentiment': torch.abs(
                    modal_cls_sentiment[i][modal1].squeeze(1) - modal_cls_sentiment[i][modal2].squeeze(1)
                )
                for modal1, modal2 in self.combine_group
            }
            for i in range(self.task_num)
        ]

        with torch.no_grad():
            for i in range(self.task_num):
                for j in range(batch_size):
                    if label[self.TASK_ID_MAP[i]][j] == 0:
                        for modal1, modal2 in self.combine_group:
                            target_bank = self.modal_inter_non_sarcasm_bank[i][f'{modal1}_{modal2}_non_sarcasm_bank']
                            if target_bank.full():
                                target_bank.get()
                            target_bank.put(modal_inter_sims[f'{modal1}_{modal2}_sims'][j])
                    elif label[self.TASK_ID_MAP[i]][j] == 1:
                        for modal1, modal2 in self.combine_group:
                            target_bank = self.modal_inter_sarcasm_bank[i][f'{modal1}_{modal2}_sarcasm_bank']
                            if target_bank.full():
                                target_bank.get()
                            target_bank.put(modal_inter_sims[f'{modal1}_{modal2}_sims'][j])

        modal_inter_lamda_semantic = [{} for _ in range(self.task_num)]
        for i in range(self.task_num):
            for modal1, modal2 in self.combine_group:
                target_non_bank = self.modal_inter_non_sarcasm_bank[i][f'{modal1}_{modal2}_non_sarcasm_bank']
                target_bank = self.modal_inter_sarcasm_bank[i][f'{modal1}_{modal2}_sarcasm_bank']

                if target_bank.full() and target_non_bank.full():
                    with torch.no_grad():
                        sarcasm_list = list(target_bank.queue)
                        mu_sarcasm = sum(sarcasm_list) / self.model_config['memory_length']
                        sigma_sarcasm = torch.sqrt(sum([(tmp - mu_sarcasm) ** 2 for tmp in sarcasm_list]))

                        non_sarcasm_list = list(target_non_bank.queue)
                        mu_non_sarcasm = sum(non_sarcasm_list) / self.model_config['memory_length']
                        sigma_non_sarcasm = torch.sqrt(sum([(tmp - mu_non_sarcasm) ** 2 for tmp in non_sarcasm_list]))

                    prob_sarcasm = (
                            (1 / (sigma_sarcasm * np.sqrt(2 * math.pi))) *
                            torch.exp(
                                -50 * ((modal_inter_sims[
                                            f'{modal1}_{modal2}_sims'] - mu_sarcasm) / sigma_sarcasm) ** 2)
                    )
                    prob_non_sarcasm = (
                            (1 / (sigma_non_sarcasm * np.sqrt(2 * math.pi))) *
                            torch.exp(-50 * ((modal_inter_sims[
                                                  f'{modal1}_{modal2}_sims'] - mu_non_sarcasm) / sigma_non_sarcasm) ** 2)
                    )
                    modal_inter_lamda_semantic[i][f'{modal1}_{modal2}_lamda_semantic'] = prob_sarcasm - prob_non_sarcasm
                else:
                    modal_inter_lamda_semantic[i][f'{modal1}_{modal2}_lamda_semantic'] = torch.zeros_like(
                        modal_inter_lamda_sentiment[i][f'{modal1}_{modal2}_lambda_sentiment']
                    )

        # fusion
        final_cls = [[] for _ in range(self.task_num)]
        for i in range(self.task_num):
            for modal1, modal2 in self.combine_group:
                # (bs, aligned_dim)
                modal1_modal2_semantic_cls = fusion(
                    semantic_modal_embeddings[modal1],  # (bs, aligned_dim)
                    semantic_modal_embeddings[modal2],  # (bs, aligned_dim)
                    self.multimodal_fusion
                )
                modal1_modal2_sentiment_cls = fusion(
                    modal_cls_sentiment_embedding[i][modal1],  # (bs, aligned_dim)
                    modal_cls_sentiment_embedding[i][modal2],  # (bs, aligned_dim)
                    self.multimodal_fusion
                )
                # (bs, 2 * aligned_dim)
                final_cls[i].append(
                    (fusion(modal1_modal2_semantic_cls, modal1_modal2_sentiment_cls, self.multilevel_fusion)))

        # (bs, modal_combine_num * 2 * aligned_dim)  -> (bs, )
        final_cls = [
            self.final_fc[i](torch.cat(final_cls[i], dim=-1)).squeeze()
            for i in range(self.task_num)
        ]

        loss = sentiment_contrast_loss + text_sentiment_loss
        record = {
            'humor_loss': 0,
            'sarcasm_loss': 0,
            'humor_label': [],
            'sarcasm_label': [],
            'humor_predict': [],
            'sarcasm_predict': [],
        }
        for i in range(self.task_num):
            # (bs, )
            current_task_predict_logits: torch.Tensor = sum([final_cls[i], -self.model_config['constant']] + [
                self.model_config['lambda_semantic'] * v
                for v in modal_inter_lamda_semantic[i].values()
            ] + [self.model_config['lambda_sentiment'] * v for v in modal_inter_lamda_sentiment[i].values()])
            current_task_loss = F.binary_cross_entropy_with_logits(current_task_predict_logits, label[self.TASK_ID_MAP[0]].float())
            loss = loss + current_task_loss if i != self.main_task_id else current_task_loss * self.main_task_weight

            current_task_predict = current_task_predict_logits.sigmoid()
            current_task_predict[current_task_predict >= 0.5] = 1
            current_task_predict[current_task_predict < 0.5] = 0

            record[self.TASK_ID_MAP[i] + '_loss'] = current_task_loss
            record[self.TASK_ID_MAP[i] + '_predict'] = current_task_predict.tolist()
            record[self.TASK_ID_MAP[i] + '_label'] = label[self.TASK_ID_MAP[i]].tolist()

        return {
            'loss': loss,
            **record
        }

    def _process_one_modal(
            self,
            input_feature: Dict[str, torch.Tensor],
            input_feature_length: Dict[str, torch.Tensor],
            label: torch.Tensor,
            text_sentiment: torch.Tensor,
            **kwargs
    ):
        batch_size = label.shape[0]

        # 模态对齐  (bs, seq/audio/visual_len, aligned_dim)
        aligned_features = self.do_aligned_features(input_feature)

        modal_cls_features = {
            modal: get_mean_by_length(modal_tensor, input_feature_length[modal])
            for modal, modal_tensor in aligned_features.items()
        }

        # semantic model, value = (bs, aligned_dim)
        semantic_modal_embeddings = {}
        for modal, modal_cls in modal_cls_features.items():
            # (aligned_dim, )
            variance = F.normalize(torch.var(modal_cls, dim=0), dim=-1)
            semantic_modal_embeddings[modal] = modal_cls + modal_cls * variance.unsqueeze(0).repeat(batch_size, 1)

        # sentiment model
        text_sentiment_loss = sum(
            self.get_sentiment_loss(text_sentiment, aligned_features, input_feature_length, i)
            for i in range(self.task_num)
        )

        # (bs, aligned_dim)  -> (bs, )
        final_cls = [
            self.final_fc(torch.cat(list(semantic_modal_embeddings.values()), dim=-1)).squeeze()
            for i in range(self.task_num)
        ]

        loss = text_sentiment_loss
        record = {
            'humor_loss': 0,
            'sarcasm_loss': 0,
            'humor_label': [],
            'sarcasm_label': [],
            'humor_predict': [],
            'sarcasm_predict': [],
        }
        for i in range(self.task_num):
            # (bs, )
            current_task_predict_logits: torch.Tensor = final_cls[i] - self.model_config['constant']
            current_task_loss = F.binary_cross_entropy_with_logits(current_task_predict_logits, label.float())
            loss = loss + current_task_loss

            current_task_predict = current_task_predict_logits.sigmoid()
            current_task_predict[current_task_predict >= 0.5] = 1
            current_task_predict[current_task_predict < 0.5] = 0

            record[self.TASK_ID_MAP[i] + '_loss'] = current_task_loss
            record[self.TASK_ID_MAP[i] + '_predict'] = current_task_predict.tolist()
            record[self.TASK_ID_MAP[i] + '_label'] = label[self.TASK_ID_MAP[i]].tolist()

        return {
            'loss': loss,
            **record
        }

    def forward(self, *args, **kwargs):
        if len(self.modals_config) >= 2:
            return self._process_more_than_one_modal(*args, **kwargs)
        else:
            return self._process_one_modal(*args, **kwargs)
