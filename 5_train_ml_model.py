import sys
from typing import List

from omegaconf import OmegaConf
from senticnet.senticnet import SenticNet
from transformers import set_seed, AutoTokenizer

from utils import ML_OPTIMIZER_FN, ML_SCHEDULER_FN, ML_TEST_FN, ML_EVAL_FN
from utils.inputter import _load_cache, CMSHDDataset
from utils.tools import build_path, calc_model_params, read_subtitle_json
from utils.trainer import Trainer, TrainingArguments

sn = SenticNet()


def get_word_level_sentiment(text: str, tokenizer):
    if tokenizer is not None:
        word_list = tokenizer.tokenize(text)
    else:
        word_list = text.split()
    text_res = []
    for word in word_list:
        try:
            word_polarity_value = float(sn.concept(word)['polarity_value'])
        except:
            word_polarity_value = float(0)
        text_res.append(word_polarity_value)
    return text_res


def load_humor_sarcasm_data(data_dir: str, used_modals: List[str], model_name: str, tokenizer=None):
    train_data, valid_data, test_data = _load_cache(data_dir)

    _d = {'train': train_data, 'valid': valid_data, 'test': test_data}
    _dataset = []
    for split in ['train', 'valid', 'test']:
        json_total, json_data = read_subtitle_json(build_path(data_dir, f'{split}.json'))
        for i in range(json_total):
            _d[split][i]['humorLabel'] = json_data[i]['isHumor']
            _d[split][i]['sarcasmLabel'] = json_data[i]['isSarcasm']
            _d[split][i].pop('label')
            if model_name == 'dip':
                assert tokenizer is not None
                sentiment = get_word_level_sentiment(json_data[i]['sentence'], tokenizer)
                _d[split][i]['sentiment'] = sentiment
        _dataset.append(
            CMSHDDataset(_d[split], used_modals, True, ['sentiment'] if model_name == 'dip' else [])
        )

    return (item for item in _dataset)


def build_model(model_name: str, modals_config: dict, model_config: dict):
    if model_name == 'mcwf':
        from models import MCWFML
        model = MCWFML(modals_config, model_config)
    elif model_name == 'cubemlp':
        from models import CubeMLPML
        model = CubeMLPML(modals_config, model_config)
    elif model_name == 'misa':
        from models import MISAML
        model = MISAML(modals_config, model_config)
    elif model_name == 'dip':
        from models import DIPML
        model = DIPML(modals_config, model_config)
    else:
        raise NotImplementedError

    return model


def main():
    # 加载配置
    config_file_path = 'config/small_model/ml/main.yaml'
    config = OmegaConf.to_container(OmegaConf.load(config_file_path), resolve=True, enum_to_str=True)
    common_config = config.get('common_config', {})
    train_config = config.get('train_config', {})
    model_config = config.get('model_config', {})

    model_config['main_task_id'] = 0 if common_config['task'] == 'humor' else 1

    train_config['output_dir'] = build_path(
        train_config['output_dir'],
        common_config['task'],
        '&'.join(common_config['used_modals'])
    )

    model_name = config['model_name']
    custom_config_file_path = build_path('config/small_model/ml', f'{model_name}.yaml')
    custom_config = OmegaConf.to_container(OmegaConf.load(custom_config_file_path), resolve=True, enum_to_str=True)

    common_config.update(custom_config.get('common_config', {}))
    train_config.update(custom_config.get('train_config', {}))
    model_config.update(custom_config.get('model_config', {}))

    print(config)

    set_seed(train_config['seed'])
    print('加载数据')
    tokenizer = None
    if model_name == 'dip':
        tokenizer = AutoTokenizer.from_pretrained(common_config['tokenizer_pretrained_path'], trust_remote_code=True)
    train_dataset, valid_dataset, test_dataset = load_humor_sarcasm_data(
        common_config['data_dir'],
        common_config['used_modals'],
        model_name,
        tokenizer
    )

    print(f'加载{model_name}模型')
    model = build_model(
        model_name=model_name,
        modals_config=train_dataset.modal_dims,
        model_config=model_config
    ).train()
    print(f'模型参数量: {calc_model_params(model)}')

    train_args = TrainingArguments(**train_config)
    Trainer(
        model=model,
        config=train_args,
        train_dataset=train_dataset,
        collate_fn=CMSHDDataset.collate_fn_for_ml,
        eval_dataset=valid_dataset,
        eval_fn=ML_EVAL_FN[model_name],
        test_dataset=test_dataset,
        test_fn=ML_TEST_FN[model_name],
        custom_optimizer=ML_OPTIMIZER_FN[model_name],
        custom_scheduler=ML_SCHEDULER_FN[model_name]
    ).train()


if __name__ == '__main__':
    main()
