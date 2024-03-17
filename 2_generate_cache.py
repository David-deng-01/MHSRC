from omegaconf import DictConfig, OmegaConf

from utils.inputter import load_data


def main(config: DictConfig):
    for task in config.tasks:
        load_data(task, config.cache_dir, config.subtitle_file_dir, config.feature_file_dir)


if __name__ == '__main__':
    config_file_path = 'config/generate_cache/main.yaml'
    main(OmegaConf.load(config_file_path))
