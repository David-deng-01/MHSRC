import json
import os
from collections import defaultdict
from typing import List, Dict

import jieba
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import set_seed

from utils.tools import get_dir_file_path, read_subtitle_json, build_path, random_choice, timestamp_to_second


def _rebuild_files(
        split_name: str,
        split_idxes: List[int],
        all_json_data: List[dict],
        all_feature: Dict[str, List[torch.Tensor]],
        config: DictConfig
):
    if len(split_idxes) == 0:
        print(f'{split_name} 无数据')
        return
    output_json_file_path = build_path(config.output_dir, f'{split_name}.json')
    output_feature_file_path = build_path(config.output_dir, f'{split_name}.pt')

    os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_feature_file_path), exist_ok=True)

    # 保存字幕
    with open(output_json_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(split_idxes),
            "data": [all_json_data[i] for i in split_idxes]
        }, f, ensure_ascii=False)
    print(f'split = {split_name} 字幕文件保存至：{output_json_file_path}')

    torch.save({
        modal: [all_feature[modal][i] for i in split_idxes]
        for modal in all_feature
    }, output_feature_file_path)
    print(f'split = {split_name} 特征文件保存至：{output_feature_file_path}')


def generate_stats(split: str, config: DictConfig):
    # 读字幕
    subtitle_file_path = build_path(config.output_dir, f'{split}.json')

    tasks = ['humor', 'sarcasm']

    record = {}

    for task in tasks:
        task_record = {
            f'is{task.title()}': {
                'cnt': 0,
                'avg_words': 0,
                'avg_duration': 0
            },
            f'non{task.title()}': {
                'cnt': 0,
                'avg_words': 0,
                'avg_duration': 0
            }
        }
        is_key = 'isHumor' if task == 'humor' else 'isSarcasm'
        for item in tqdm(read_subtitle_json(subtitle_file_path)[1], desc=f'{split}_{task}', dynamic_ncols=True):
            words = list(jieba.cut(item['sentence']))
            if item[is_key] == 0:
                task_record[f'non{task.title()}']['cnt'] += 1
                task_record[f'non{task.title()}']['avg_words'] += len(words)
                task_record[f'non{task.title()}']['avg_duration'] += timestamp_to_second(
                    item['end']) - timestamp_to_second(item['start'])
            else:
                task_record[is_key]['cnt'] += 1
                task_record[is_key]['avg_words'] += len(words)
                task_record[is_key]['avg_duration'] += timestamp_to_second(item['end']) - timestamp_to_second(
                    item['start'])

        task_record[f'non{task.title()}']['avg_words'] /= task_record[f'non{task.title()}']['cnt']
        task_record[f'non{task.title()}']['avg_duration'] /= task_record[f'non{task.title()}']['cnt']

        task_record[is_key]['avg_words'] /= task_record[is_key]['cnt']
        task_record[is_key]['avg_duration'] /= task_record[is_key]['cnt']

        record[task] = task_record

    with open(build_path(config.output_dir, f'{split}_stat.json'), 'w', encoding='utf-8') as f:
        json.dump(record, f, ensure_ascii=False, indent=4)


def main(config: DictConfig):
    set_seed(config.seed)
    # 所有的字幕路径, rebuild之后的
    json_file_paths = get_dir_file_path(
        dir_name=config.json_file_dir,
        skip_file_names=[f'{s_e}.json' for s_e in config.skip_season_eps]
    )
    # 存储所有模态的数据
    all_modals_feature = defaultdict(list)
    all_json_data = []

    idx, idx_to_season_episode = 0, dict()
    # {'speaker': {'0_0': [], '1_0': [], '0_1': [], '1_1': []}, ...}
    group_by_speaker = defaultdict(dict)

    for j_f_p in tqdm(json_file_paths, desc='读数据', dynamic_ncols=True):
        s_e = os.path.splitext(os.path.basename(j_f_p))[0]

        feature_file_path = build_path(config.feature_file_dir, f'{s_e}.pt')
        feature_data = torch.load(feature_file_path, map_location='cpu')
        for modal, v in feature_data.items():
            all_modals_feature[modal].extend(v)

        json_total, data = read_subtitle_json(j_f_p)

        modals_cnt = [len(v) for v in feature_data.values()]
        assert len(set(modals_cnt + [json_total])) == 1, f's_e={s_e}, modals_cnt={modals_cnt}, json_total={json_total}'

        for item in data:
            is_humor, is_sarcasm = item['isHumor'], item['isSarcasm']
            _key = f'{is_humor}_{is_sarcasm}'
            if _key not in group_by_speaker[item['speaker']]:
                group_by_speaker[item['speaker']][_key] = []
            group_by_speaker[item['speaker']][_key].append(idx)
            idx_to_season_episode[idx] = s_e
            idx += 1
        all_json_data.extend(data)

    # 按照 speaker, 分割数据
    train_idxes, valid_idxes, test_idxes = [], [], []
    for speaker in group_by_speaker:
        for label_key, idx_list in group_by_speaker[speaker].items():
            _size = len(idx_list)
            train_size = int(_size * config.train_ratio)
            valid_size = (_size - train_size) // 2

            _train_idxes = random_choice(idx_list, train_size)
            _valid_idxes = random_choice([i for i in idx_list if i not in _train_idxes], valid_size)
            _test_idxes = [i for i in idx_list if i not in _train_idxes and i not in _valid_idxes]
            train_idxes.extend(_train_idxes)
            valid_idxes.extend(_valid_idxes)
            test_idxes.extend(_test_idxes)
    # 根据 train_idxes, valid_idxes, test_idxes 整合 json 和 特征
    _rebuild_files('train', train_idxes, all_json_data, all_modals_feature, config)
    _rebuild_files('valid', valid_idxes, all_json_data, all_modals_feature, config)
    _rebuild_files('test', test_idxes, all_json_data, all_modals_feature, config)

    # 生成统计数据
    generate_stats('train', config)
    generate_stats('valid', config)
    generate_stats('test', config)


if __name__ == '__main__':
    config_file_path = 'config/split_data/main.yaml'
    main(OmegaConf.load(config_file_path))
