import os

from omegaconf import OmegaConf

from .driver import PROJECT_ROOT_PATH


def get_absolute_file_path(file_path):
    if file_path.startswith("/"):
        return file_path
    else:
        return os.path.join(PROJECT_ROOT_PATH, file_path)


def merge_config(front_config, back_config):
    return OmegaConf.merge(front_config, back_config)


def init_experiment_config(config_path):
    # 读config的逻辑是，从configs/目录开始，读取base.yaml然后跳到下一个目录，如果有base就要读base,一直督读到指定的config，后一个config开源覆盖前面config的内容
    if not config_path.startswith("/"):
        config_path = get_absolute_file_path(config_path)

    current_path = get_absolute_file_path(config_path)
    # check if exist base.yaml
    if not os.path.isfile(current_path):
        raise ValueError("config file {} is not a supported file".format(current_path))

    while os.path.normpath(current_path) != os.path.normpath(PROJECT_ROOT_PATH):
        if os.path.isfile(current_path):
            config = OmegaConf.load(current_path)
            # print("load config from {}".format(current_path))
        elif os.path.isdir(current_path):
            base_config_path = os.path.join(current_path, "base.yaml")
            if os.path.exists(base_config_path):
                base_config = OmegaConf.load(base_config_path)
                # print("load base config from {}".format(base_config_path))
                config = OmegaConf.merge(base_config, config)
        else:
            # print("path {} has no base.yaml".format(current_path))
            pass
        current_path = os.path.dirname(current_path)
    return config


def _set_nested_value(d, keys, value):
    if len(keys) == 1:
        d[keys[0]] = value
    else:
        nested_dict = d.get(keys[0], {})
        nested_dict = _set_nested_value(nested_dict, keys[1:], value)
        d[keys[0]] = nested_dict
    return d


def override_phase_config(config):
    phase = config.phase
    phase_config = config.get(phase, {})
    config = OmegaConf.merge(config, phase_config)
    return config


def parser_override_args(config, kwargs):
    # "a.b=True" to {"a": {"b": True}} and process nesting, transfer b to real value
    new_kwargs = OmegaConf.from_dotlist(kwargs)
    # override config with new_kwargs
    config = OmegaConf.merge(config, new_kwargs)
    OmegaConf.resolve(config)
    return config
