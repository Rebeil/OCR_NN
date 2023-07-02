import yaml


def load_settings(path: str = '/home/rain/vs_code/relize/yaml_config/config_main.yml') -> dict:
    option_path = path
    with open(option_path, 'r') as file_option:
        option = yaml.safe_load(file_option)
    return option
