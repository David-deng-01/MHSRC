import sys
from typing import List

from omegaconf import OmegaConf
from senticnet.senticnet import SenticNet
from tqdm import tqdm
from transformers import AutoTokenizer

from utils import CLASSIFIER_EVAL_FN, CLASSIFIER_TEST_FN, OPTIMIZER_FN, SCHEDULER_FN
from utils.inputter import _load_cache, CMSHDDataset
from utils.tools import calc_model_params, build_path, read_subtitle_json
from utils.trainer import TrainingArguments, Trainer

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


def convert_multilabel_to_list(labels: List[int], max_types: int):
    """
    原始类标签：[[], []]
    展平: sum(原始类别标签, [])
    Args:
        max_types:
        labels:
    """
    result = [0] * max_types
    for i in labels:
        result[i] = 1
    return result


def load_data(task: str, data_dir: str, used_modals: List[str], model_name: str, tokenizer=None):
    _type_key = 'humorType' if task == 'humor' else 'sarcasmType'
    _max_types = 6 if task == 'sarcasm' else 8

    train_data, valid_data, test_data = _load_cache(data_dir)

    train_total, train_json = read_subtitle_json(build_path(data_dir, 'train.json'))
    valid_total, valid_json = read_subtitle_json(build_path(data_dir, 'valid.json'))
    test_total, test_json = read_subtitle_json(build_path(data_dir, 'test.json'))

    if model_name == 'dip':
        assert tokenizer is not None
        for i in tqdm(range(train_total), total=train_total, desc='处理train词极性'):
            sentiment = get_word_level_sentiment(train_json[i]['sentence'], tokenizer)
            train_data[i]['sentiment'] = sentiment
        for i in tqdm(range(valid_total), total=valid_total, desc='处理valid词极性'):
            sentiment = get_word_level_sentiment(valid_json[i]['sentence'], tokenizer)
            valid_data[i]['sentiment'] = sentiment
        for i in tqdm(range(test_total), total=test_total, desc='处理test词极性'):
            sentiment = get_word_level_sentiment(test_json[i]['sentence'], tokenizer)
            test_data[i]['sentiment'] = sentiment

    for i in tqdm(range(train_total), total=train_total, desc='处理 train label'):
        train_data[i]['label'] = convert_multilabel_to_list(
            list(set(sum(train_json[i][_type_key], []))),
            _max_types
        )
        train_data[i]['raw_label'] = train_json[i][_type_key]

    for i in tqdm(range(valid_total), total=valid_total, desc='处理 valid label'):
        valid_data[i]['label'] = convert_multilabel_to_list(
            list(set(sum(valid_json[i][_type_key], []))),
            _max_types
        )
        valid_data[i]['raw_label'] = valid_json[i][_type_key]

    for i in tqdm(range(test_total), total=test_total, desc='处理 test label'):
        test_data[i]['label'] = convert_multilabel_to_list(
            list(set(sum(test_json[i][_type_key], []))),
            _max_types
        )
        test_data[i]['raw_label'] = test_json[i][_type_key]

    train_dataset = CMSHDDataset(train_data, used_modals, False, ['sentiment', 'raw_label'] if model_name == 'dip' else ['raw_label'])
    valid_dataset = CMSHDDataset(valid_data, used_modals, False, ['sentiment', 'raw_label'] if model_name == 'dip' else ['raw_label'])
    test_dataset = CMSHDDataset(test_data, used_modals, False, ['sentiment', 'raw_label'] if model_name == 'dip' else ['raw_label'])

    return train_dataset, valid_dataset, test_dataset


def build_model(model_name: str, modals_config: dict, model_config: dict):
    if model_name == 'mcwf':
        from models import MCWF
        model = MCWF(modals_config, model_config)
    elif model_name == 'misa':
        from models import MISA
        model = MISA(modals_config, model_config)
    elif model_name == 'cubemlp':
        from models import CubeMLP
        model = CubeMLP(modals_config, model_config)
    else:
        raise NotImplementedError

    return model


def main():
    config_file_path = 'config/type_classifier/main.yaml'
    config = OmegaConf.to_container(OmegaConf.load(config_file_path), enum_to_str=True, resolve=True)
    common_config = config.get('common_config', {})
    train_config = config.get('train_config', {})
    model_config = config.get('model_config', {})

    # 单任务幽默讽刺检测：output/small_model
    # 多任务：output/small_model/ml
    # 分类任务：output/small_model/classifier
    train_config['output_dir'] = build_path(
        train_config['output_dir'],
        "classifier",
        common_config['task'],
        '&'.join(common_config['used_modals'])
    )

    model_name = config['model_name']
    custom_file_path = f'config/type_classifier/{model_name}.yaml'
    custom_config = OmegaConf.to_container(OmegaConf.load(custom_file_path), enum_to_str=True, resolve=True)
    custom_config['model_config']['num_class'] = 6 if common_config['task'] == 'sarcasm' else 8

    common_config.update(custom_config.get('common_config', {}))
    train_config.update(custom_config.get('train_config', {}))
    model_config.update(custom_config.get('model_config', {}))

    print(config)
    print('加载数据')
    tokenizer = None
    if model_name == 'dip':
        tokenizer = AutoTokenizer.from_pretrained(common_config['tokenizer_pretrained_path'], trust_remote_code=True)
    train_dataset, valid_dataset, test_dataset = load_data(
        task=common_config['task'],
        data_dir=common_config['data_dir'],
        used_modals=common_config['used_modals'],
        model_name=model_name,
        tokenizer=tokenizer
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
        collate_fn=CMSHDDataset.collate_fn,
        eval_dataset=valid_dataset,
        eval_fn=CLASSIFIER_EVAL_FN[model_name],
        test_dataset=test_dataset,
        test_fn=CLASSIFIER_TEST_FN[model_name],
        custom_optimizer=OPTIMIZER_FN[model_name],
        custom_scheduler=SCHEDULER_FN[model_name]
    ).train()


if __name__ == '__main__':
    main()
