import json
import os
from collections import defaultdict
from typing import List, Dict

import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from transformers import set_seed

from utils.tools import encode
from utils.tools import read_subtitle_json, get_dir_file_path, build_path, random_choice


def _split_idxes(idxes: List[int], ratio: List[float]):
    _size = len(idxes)
    if _size <= 2:
        return idxes, [], []
    elif _size <= 10:
        valid_idxes = random_choice(idxes, 1)
        test_idxes = random_choice([i for i in idxes if i not in valid_idxes], 1)
        train_idxes = list(filter(lambda i: i not in test_idxes and i not in valid_idxes, idxes))
        return train_idxes, valid_idxes, test_idxes
    else:
        valid_idxes = random_choice(idxes, int(_size * ratio[1]))
        test_idxes = random_choice([i for i in idxes if i not in valid_idxes], int(_size * ratio[-1]))
        train_idxes = list(filter(lambda i: i not in test_idxes and i not in valid_idxes, idxes))
        return train_idxes, valid_idxes, test_idxes


def _rebuild_files(
        task: str,
        split_name: str,
        split_idxes: List[int],
        all_json_data: List[dict],
        all_feature_data: List[Dict[str, torch.Tensor]],
        config: DictConfig
):
    if len(split_idxes) == 0:
        print(f'{split_name} 无数据')
        return
    output_json_file_path = build_path(config.output_dir, task, f'{split_name}.json')
    output_feature_file_path = build_path(config.output_dir, task, f'{split_name}.pt')

    os.makedirs(os.path.dirname(output_json_file_path), exist_ok=True)
    os.makedirs(os.path.dirname(output_feature_file_path), exist_ok=True)

    # 保存字幕
    with open(output_json_file_path, 'w', encoding='utf-8') as f:
        json.dump({
            "total": len(split_idxes),
            "data": [all_json_data[i] for i in split_idxes]
        }, f, ensure_ascii=False)
    print(f'split = {split_name} 字幕文件保存至：{output_json_file_path}')

    torch.save([all_feature_data[i] for i in split_idxes], output_feature_file_path)
    print(f'split = {split_name} 特征文件保存至：{output_feature_file_path}')


def generate_cache(config: DictConfig, task: str):
    if task.lower() == 'humor':
        _is_key = 'isHumor'
        _type_key = 'humorType'
    else:
        _is_key = 'isSarcasm'
        _type_key = 'sarcasmType'

    # 读数据
    all_json_data, all_feature_data = [], []
    type_record = defaultdict(list)
    json_file_paths = get_dir_file_path(
        dir_name=config.json_file_dir,
        file_ext=['.json'],
        skip_file_names=[f'{s_e}.json' for s_e in config.skip_season_eps]
    )
    idx = 0
    for j_f_p in tqdm(json_file_paths, dynamic_ncols=True):
        # 当前字幕文件的特征文件
        s_e = os.path.splitext(os.path.basename(j_f_p))[0]
        feature_file_path = build_path(config.feature_file_dir, s_e + '.pt')

        # 读数据
        json_total, json_data = read_subtitle_json(j_f_p)
        feature_data = torch.load(feature_file_path)
        # feature_data 格式处理
        assert len(feature_data['text']) == len(feature_data['visual']) == len(feature_data['audio']) == json_total, \
            f's_e = {s_e}, 特征 = {feature_file_path}、字幕 = {j_f_p} 总量不一致'

        for i in range(json_total):
            if json_data[i][_is_key] == 1:
                # 数据、类别、特征记录
                all_json_data.append(json_data[i])
                all_feature_data.append({
                    modal: feature_data[modal][i]
                    for modal in ['visual', 'audio', 'text']
                })
                type_record[encode(sum(json_data[i][_type_key], []))].append(idx)
                idx += 1

    del feature_data
    del json_total
    del json_data

    all_train_idxes, all_valid_idxes, all_test_idxes = [], [], []
    record_to_log = {'train': defaultdict(int), 'valid': defaultdict(int), 'test': defaultdict(int)}
    for _type, _type_idxes in type_record.items():
        train_idxes, valid_idxes, test_idxes = _split_idxes(
            _type_idxes, [config.train_ratio, config.valid_ratio, config.test_ratio]
        )
        all_train_idxes.extend(train_idxes)
        all_valid_idxes.extend(valid_idxes)
        all_test_idxes.extend(test_idxes)

        record_to_log['train'][_type] += len(train_idxes)
        record_to_log['valid'][_type] += len(valid_idxes)
        record_to_log['test'][_type] += len(test_idxes)

    print('train size:', len(all_train_idxes))
    print('valid size:', len(all_valid_idxes))
    print('test size:', len(all_test_idxes))
    # 根据筛选出来的idx提取特征数据并保存
    _rebuild_files(task, 'train', all_train_idxes, all_json_data, all_feature_data, config)
    _rebuild_files(task, 'valid', all_valid_idxes, all_json_data, all_feature_data, config)
    _rebuild_files(task, 'test', all_test_idxes, all_json_data, all_feature_data, config)

    record_output_file_path = build_path(config.output_dir, task, 'type_dist_record.json')
    os.makedirs(os.path.dirname(record_output_file_path), exist_ok=True)
    with open(record_output_file_path, 'w', encoding='utf-8') as f:
        json.dump(record_to_log, f, indent=4, ensure_ascii=False)
    print('类别分布文件保存至: {}'.format(record_output_file_path))


def main():
    config_file_path = 'config/type_classifier/generate_cache.yaml'
    config = OmegaConf.load(config_file_path)
    set_seed(config.seed)
    for task in config.tasks:
        print(f'处理{task}的分类缓存数据')
        generate_cache(config, task)


if __name__ == '__main__':
    main()
