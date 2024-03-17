import json
import os
import random
import re
from typing import List, Any, Dict, Tuple

import cv2
import numpy as np
import torch.nn as nn
from torch import Tensor


def build_path(*args, is_abs=False):
    size = len(args)
    p = str(args[0])
    for pp in range(1, size):
        p = os.path.join(p, str(args[pp]))
    if is_abs:
        return os.path.abspath(p)
    return p


def is_number(s):
    try:
        int(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def get_dir_file_path(
        dir_name: str,
        file_ext: List[str] = None,
        skip_dir_names: List[str] = None,
        skip_file_names: List[str] = None,
        is_abs: bool = False
):
    if is_abs:
        dir_name = os.path.abspath(dir_name)
    if file_ext is None:
        file_ext = []
    if skip_dir_names is None:
        skip_dir_names = []
    if skip_file_names is None:
        skip_file_names = []
    # 获得所有的文件夹、文件
    arr = []
    all_file_and_dir_name = os.listdir(dir_name)
    for file_or_dir in all_file_and_dir_name:
        full_path = os.path.join(dir_name, file_or_dir)
        # 如果是目录, 递归
        if os.path.isdir(full_path):
            if file_or_dir in skip_dir_names:
                continue
            arr.extend(get_dir_file_path(full_path, file_ext, skip_dir_names, skip_file_names, is_abs))
        else:  # 如果是文件
            if file_or_dir in skip_file_names:
                continue
            if len(file_ext) > 0:
                if any(full_path.endswith(ext) for ext in file_ext):
                    arr.append(full_path)
            else:
                arr.append(full_path)
    return arr


def time_to_ms(time_str: str) -> int:
    try:
        h, m, s = time_str.split(':')
        h, m, s = h.strip(), m.strip(), s.strip()
        s, ms = re.split(r'[.,。，]', s)
        s, ms = s.strip(), ms.strip()
        assert len(h) == 2 and len(m) == 2 and len(s) == 2 and len(ms) == 3
        return int(h) * 3600000 + int(m) * 60000 + int(s) * 1000 + int(ms)
    except Exception as e:
        raise ValueError('时间格式错误！')


def ms_time_to_format(ms_time: int) -> str:
    h = ms_time // 3600000
    m = (ms_time - h * 3600000) // 60000
    s = (ms_time - h * 3600000 - m * 60000) // 1000
    ms = ms_time - h * 3600000 - m * 60000 - s * 1000

    if h < 10:
        h = f'0{h}'
    if m < 10:
        m = f'0{m}'
    if s < 10:
        s = f'0{s}'
    if ms < 10:
        ms = f'00{ms}'
    elif ms < 100:
        ms = f'0{ms}'

    return f'{h}:{m}:{s},{ms}'


def get_season_episode_from_path(f_p: str):
    # data/humor/第2季 第2期/xxx.json
    season, episode = os.path.basename(os.path.dirname(f_p)).split()
    season = int(season[1:-1])
    episode = int(episode[1:-1])
    return season, episode


def timestamp_to_second(timestamp: str = '00:00:00,000') -> float:
    """
    时间戳转换成秒
    :param timestamp: 时间戳, 格式: '00:00:00,000'
    :return: 秒
    """
    return sum(x * float(t) for x, t in zip([3600, 60, 1, 0.001], re.split('[，,.。：:；;]', timestamp)))


def random_choice(arr: List[Any], n: int = 1) -> List[Any]:
    """
    随机从 arr 中选取 n 张图片
    :param arr:
    :param n:
    :return: 选取到的图片数量
    """
    return random.sample(arr, min(n, len(arr)))


def do_image_crop(image, left: float, right: float, top: float, bottom: float):
    original_width, original_height = image.size

    crop_left = original_width * left
    crop_right = original_width * right

    crop_top = original_height * top
    crop_bottom = original_height * bottom

    cropped_img = image.crop((crop_left, crop_top, crop_right, crop_bottom))

    return cropped_img


def read_subtitle_json(f_p: str):
    with open(f_p, 'r', encoding='utf-8') as f:
        data = json.load(f)['data']
    return len(data), data


def calc_model_params(model: nn.Module):
    model_parameters = list(filter(lambda p: p.requires_grad, model.parameters()))
    total_params = sum([np.prod(p.size()) for p in model_parameters])
    return total_params


def data_2_device(data, device):
    if isinstance(data, dict):
        return {k: data_2_device(v, device) for k, v in data.items()}
    elif isinstance(data, Tensor):
        return data.to(device)
    elif isinstance(data, list):
        return [data_2_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(data_2_device(item, device) for item in data)
    else:
        return data


def rm_file(path: str):
    if os.path.exists(path) and os.path.isfile(path):
        os.remove(path)


def rm_dir(path: str):
    if os.path.exists(path) and os.path.isdir(path):
        for f_name in os.listdir(path):
            sub_path = build_path(path, f_name)
            if os.path.isdir(sub_path):
                rm_dir(sub_path)
            else:
                rm_file(sub_path)
        os.rmdir(path)


def convert_path_to_gbk(file_path: str):
    return file_path.encode('gbk').decode('gbk')


def cv_imread(file_path: str):
    return cv2.imdecode(np.fromfile(file_path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)


def cv_imwrite(img: np.ndarray, file_path: str):
    ext = os.path.splitext(os.path.basename(file_path))[1]
    cv2.imencode(ext, img)[1].tofile(file_path)


def make_infinite(dataloader):
    while True:
        for x in dataloader:
            yield x


def encode(int_list: List[int] = None):
    if int_list is None:
        int_list = []

    if len(int_list) == 0:
        return 0

    r = 0
    for bit in int_list:
        r += 1 << bit

    return r


def c_n_m(arr_list: List[Any], m: int, n: int = None) -> List[Tuple[Any, ...]]:
    """
    C_n^m
    Args:
        arr_list: sample arr
        m (int): the number of samples needed to sample.
        n (int): you don't need to provide n. n is equal to len(arr_list).
                If you provide a specific n, arr_list would be equal to arr_list[:n]
    """

    def _fun(arr: List[Any], left: int, right: int, select_cnt: int):
        """
        Args:
            arr: need to be sampled
            left: left pointer, included
            right: right pointer, not included
            select_cnt: the number of samples
        """
        # some condition need to consider
        result = []
        for i in range(left, right + 1 - select_cnt):
            if select_cnt - 1 > 0:
                combine_result = _fun(arr, i + 1, right, select_cnt - 1)
                result.extend([(arr[i],) + item for item in combine_result])
            else:
                result.append((arr[i],))
        return result

    if n is None:
        n = len(arr_list)
    arr_list = arr_list[:n]
    return _fun(arr_list, 0, n, m)

def config_to_dict(args) -> Dict[str, Any]:
    config_class = type(args)
    class_attrs = {k: v for k, v in vars(config_class).items() if not k.startswith('__')}
    instance_attrs = vars(args)
    d = {**class_attrs}
    d.update(**instance_attrs)
    return d


def do_shuffle(arr, shuffle_times: int = 1):
    for _ in range(shuffle_times):
        random.shuffle(arr)
