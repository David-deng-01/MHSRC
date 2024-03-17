import json
import os
from typing import List, Dict, Callable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from .tools import build_path, random_choice, read_subtitle_json


def _load_cache(true_cache_dir):
    train_cache_file_path = build_path(true_cache_dir, 'train.pt')
    valid_cache_file_path = build_path(true_cache_dir, 'valid.pt')
    test_cache_file_path = build_path(true_cache_dir, 'test.pt')

    assert os.path.exists(train_cache_file_path)
    assert os.path.exists(valid_cache_file_path)
    assert os.path.exists(test_cache_file_path)

    train_data = torch.load(train_cache_file_path)
    valid_data = torch.load(valid_cache_file_path)
    test_data = torch.load(test_cache_file_path)

    return train_data, valid_data, test_data


def _balance_upper(is_key_idxes: List[int], not_is_key_idxes: List[int]):
    """过采样, train, 增加 is_key 的含量"""
    dis = len(not_is_key_idxes) - len(is_key_idxes)

    additional_is_key_idxes = []
    while dis > 0:
        _d = random_choice(is_key_idxes, dis)
        additional_is_key_idxes.extend(_d)
        dis -= len(_d)
    return is_key_idxes + additional_is_key_idxes + not_is_key_idxes


def _balance_lower(is_key_idxes: List[int], not_is_key_idxes: List[int]):
    """负采样, valid, test, 减少 not_is_key 含量 """
    dis = len(not_is_key_idxes) - len(is_key_idxes)

    if dis < 0:
        return random_choice(is_key_idxes, abs(dis)) + not_is_key_idxes
    elif dis > 0:
        return is_key_idxes + random_choice(not_is_key_idxes, abs(dis))
    else:
        return is_key_idxes + not_is_key_idxes


def load_data(
        task: str,
        cache_dir: str,
        subtitle_file_dir: str,
        feature_file_dir: str
):
    SPLITS = ['train', 'valid', 'test']
    true_cache_dir = os.path.join(cache_dir, task)
    os.makedirs(true_cache_dir, exist_ok=True)
    try:
        return _load_cache(true_cache_dir)
    except Exception as e:
        print('缓存文件不存在, 正在生成缓存文件……')

    IS_KEY = 'isHumor' if task.lower() == 'humor' else 'isSarcasm'
    for split in SPLITS:
        json_file_path = os.path.join(subtitle_file_dir, split + '.json')
        feature_file_path = os.path.join(feature_file_dir, split + '.pt')

        split_subtitle_total, split_subtitle_data = read_subtitle_json(json_file_path)
        # {'visual': [], 'audio': [], 'text': []}
        _split_feature = torch.load(feature_file_path, map_location='cpu')
        assert len(set([len(v) for v in _split_feature.values()] + [split_subtitle_total])) == 1
        # [{'visual': '', 'audio': '', 'text': ''}, ...]
        split_feature = []
        for i in range(split_subtitle_total):
            split_feature.append({
                'visual': _split_feature['visual'][i],
                'text': _split_feature['text'][i],
                'audio': _split_feature['audio'][i],
            })

        is_key_idxes = [idx for idx, item in enumerate(split_subtitle_data) if item[IS_KEY] == 1]
        not_is_key_idxes = [idx for idx, item in enumerate(split_subtitle_data) if item[IS_KEY] == 0]

        _func = _balance_upper if split == 'train' else _balance_lower
        cache_idxes = _func(is_key_idxes, not_is_key_idxes)

        feature_split_cache = []
        subtitle_split_cache = []
        for _idx in cache_idxes:
            subtitle_split_cache.append(split_subtitle_data[_idx])
            feature_split_cache.append({
                'label': split_subtitle_data[_idx][IS_KEY],
                **split_feature[_idx]
            })
        torch.save(feature_split_cache, build_path(true_cache_dir, split + '.pt'))
        with open(build_path(true_cache_dir, split + '.json'), 'w', encoding='utf-8') as f:
            json.dump({
                'total': len(subtitle_split_cache),
                'data': subtitle_split_cache
            }, f, ensure_ascii=False)

        print(f'{split}缓存文件保存至：{true_cache_dir}')
        del feature_split_cache
        del split_feature
        del split_subtitle_data
        del cache_idxes
        del is_key_idxes
        del not_is_key_idxes
    return _load_cache(true_cache_dir)


class CMSHDDataset(Dataset):
    def __init__(
            self,
            data_list: List[dict],
            modals: List[str],
            for_ml: bool = False,
            addition_key: List[str] = None
    ):
        super(CMSHDDataset, self).__init__()
        # data_list = [{'visual': '', 'audio': '', 'text': ''}, ...]
        self.data = data_list
        self.length = len(data_list)
        self.modals = modals
        self.label_keys = ['label'] if not for_ml else ['humorLabel', 'sarcasmLabel']
        if addition_key is None:
            addition_key = []
        self.addition_key = addition_key

    @property
    def modal_dims(self) -> Dict[str, int]:
        return {
            modal: self.data[0][modal].shape[-1]
            for modal in self.modals
        }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return_dict = {
            modal: self.data[idx][modal]
            for modal in self.modals
        }
        for _k in self.label_keys:
            return_dict[_k] = self.data[idx][_k]
        for _k in self.addition_key:
            return_dict[_k] = self.data[idx][_k]
        return return_dict

    @staticmethod
    def collate_fn(batch):
        feature_dict = {}
        feature_len_dict = {}
        y, text_sentiment = [], []
        for item in batch:
            for modal in item:  # modal in [label, visual, audio, text]
                if modal == 'label':
                    y.append(item[modal])
                elif modal == 'sentiment':
                    text_sentiment.append(torch.FloatTensor(item[modal]))
                else:
                    assert item[modal].dim() == 2
                    if modal not in feature_dict:
                        feature_dict[modal] = []
                    if modal not in feature_len_dict:
                        feature_len_dict[modal] = []

                    feature_dict[modal].append(item[modal])
                    feature_len_dict[modal].append(item[modal].shape[0])

        return {
            'input_feature': {
                modal: pad_sequence(feature_dict[modal], batch_first=True)
                for modal in feature_dict
            },
            'input_feature_length': {
                k: torch.LongTensor(v)
                for k, v in feature_len_dict.items()
            },
            'label': torch.LongTensor(y),
            'batch_size': len(y),
            'text_sentiment': pad_sequence(text_sentiment, batch_first=True) if len(text_sentiment) > 0 else None,
        }

    @staticmethod
    def collate_fn_for_ml(batch):
        feature_dict = {}
        feature_len_dict = {}
        humor_label, sarcasm_label, text_sentiment = [], [], []
        for item in batch:
            for modal in item:  # modal in [humorLabel, sarcasmLabel, visual, audio, text, task_id]
                if modal == 'humorLabel':
                    humor_label.append(item[modal])
                elif modal == 'sarcasmLabel':
                    sarcasm_label.append(item[modal])
                elif modal == 'sentiment':
                    text_sentiment.append(torch.FloatTensor(item[modal]))
                else:
                    assert item[modal].dim() == 2
                    if modal not in feature_dict:
                        feature_dict[modal] = []
                    feature_dict[modal].append(item[modal])

                    if modal not in feature_len_dict:
                        feature_len_dict[modal] = []
                    feature_len_dict[modal].append(item[modal].shape[0])
        return {
            'input_feature': {
                modal: pad_sequence(feature_dict[modal], batch_first=True)
                for modal in feature_dict
            },
            'input_feature_length': {
                k: torch.LongTensor(v)
                for k, v in feature_len_dict.items()
            },
            'label': {
                'humor': torch.LongTensor(humor_label),
                'sarcasm': torch.LongTensor(sarcasm_label)
            },
            'batch_size': len(batch),
            'text_sentiment': pad_sequence(text_sentiment, batch_first=True) if len(text_sentiment) > 0 else None
        }


def create_dataloader(dataset, batch_size, collate_fn: Callable = None, shuffle: bool = False):
    return DataLoader(
        dataset=dataset,
        drop_last=False,
        collate_fn=collate_fn,
        shuffle=shuffle,
        batch_size=batch_size
    )
