"""
此文件会用到8_train_small_model.py生成的缓存文件
因此，必须在在8_train_small_model.py之后执行
"""
import os
import sys
from urllib.parse import quote

from PIL import Image
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from models import ChatQwenVL, ChatGPT4V
from utils.tools import build_path, read_subtitle_json, get_season_episode_from_path


def generate_cache(config: DictConfig):
    print('挑选图片生成缓存文件中')
    for task in config.cache_task:
        # 打开 json 缓存文件
        data = read_subtitle_json(build_path(config.json_file_dir, task, config.target_split + '.json'))[1]

        for item in tqdm(data, total=len(data), desc=task):
            src_path = item['image_path']
            assert os.path.exists(src_path)
            # src_path = output/rebuild/crop_images/s_e/subtitleNo.ong
            s_e = config.season_episode_format.format(*get_season_episode_from_path(src_path))

            dst_path = build_path(config.output_image_file_dir, s_e, os.path.basename(src_path))
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            Image.open(src_path).save(dst_path)
        print(f'task = {task} 完成')


def do_run(config: DictConfig):
    model_name = config.model_name

    output_dir = build_path(config.output_dir, config.task, config.used_modals)
    os.makedirs(output_dir, exist_ok=True)

    # 读取提示模板
    prompt_template = config['prompt_template'][config.task][config.used_modals]
    if model_name == 'qwen':
        prompt_template = prompt_template.replace('\n图片：{}', '')

    # 读字幕.json
    json_data = read_subtitle_json(
        build_path(config.json_file_dir, config.task, config.target_split + '.json')
    )[1]
    image_path_list = []

    for item in json_data:
        image_path = item['image_path'].replace('output/rebuild/images/', '')
        if config.use_local_image:
            new_image_path = 'file:\\' + os.path.abspath(
                build_path(config.local_image_dir, image_path)
            )
        else:
            new_image_path = build_path(config.network_image_prefix, quote(image_path))
        image_path_list.append(new_image_path)
    assert len(image_path_list) == len(json_data)

    # 创建模型
    if config.model_name == 'qwen':
        model_config = OmegaConf.load(build_path('config/llm_baseline', f'{config.model_name}.yaml'))
        model = ChatQwenVL(
            task=config.task,
            prompt_template=prompt_template,
            subtitle_list=json_data,
            image_path_list=image_path_list,
            used_modals=config.used_modals,
            max_workers=None if config.workers == 'default' else config.workers,
            output_dir=output_dir,
            **model_config
        )
    elif config.model_name == 'gpt':
        model_config = OmegaConf.load(build_path('config/llm_baseline', f'{config.model_name}.yaml'))
        model = ChatGPT4V(
            task=config.task,
            prompt_template=prompt_template,
            subtitle_list=json_data,
            image_path_list=image_path_list,
            used_modals=config.used_modals,
            max_workers=None if config.workers == 'default' else config.workers,
            output_dir=output_dir,
            **model_config
        )
    else:
        raise NotImplementedError
    model.chat()


def main(config: DictConfig):
    if config.generate_cache:
        generate_cache(config)
        print(f'大模型需要的图片选择完成, 将{config.output_image_file_dir}上传到图床后')
        print('完成后, 修改配置文件将`generate_cache`改为`false`')
    if config.do_run:
        do_run(config)


if __name__ == '__main__':
    config_file_path = 'config/llm_baseline/main.yaml'
    config = OmegaConf.load(config_file_path)
    model_config = OmegaConf.load(build_path('config/llm_baseline', config.model_name + '.yaml'))
    config.model_config = model_config

    if config.generate_cache or config.do_run:
        if config.generate_cache:
            config.do_run = False
        else:
            config.generate_cache = False
    if len(sys.argv) >= 2:
        config.task = sys.argv[1]
        if len(sys.argv) >= 3:
            config.model_name = sys.argv[2]
            if len(sys.argv) == 4:
                config.used_modals = sys.argv[3]
    print(f'task: {config.task}, USED_MODALS = {config.used_modals}, CURRENT_LLM = {config.model_name}')
    main(config)
