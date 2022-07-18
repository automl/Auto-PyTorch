import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace


def read_return_initial_configurations(
    config_space: ConfigurationSpace,
    portfolio_selection: str
) -> List[Configuration]:
    # read and validate initial configurations
    portfolio_path = portfolio_selection if portfolio_selection != "greedy" else \
        os.path.join(os.path.dirname(__file__), '../configs/greedy_portfolio.json')
    try:
        initial_configurations_dict: List[Dict[str, Any]] = json.load(open(portfolio_path))
    except FileNotFoundError:
        raise FileNotFoundError("The path: {} provided for 'portfolio_selection' for "
                                "the file containing the portfolio configurations "
                                "does not exist. Please provide a valid path".format(portfolio_path))
    initial_configurations: List[Configuration] = list()
    for configuration_dict in initial_configurations_dict:
        try:
            configuration = Configuration(config_space, configuration_dict)
            initial_configurations.append(configuration)
        except Exception as e:
            warnings.warn(f"Failed to convert {configuration_dict} into"
                          f" a Configuration with error {e}. "
                          f"Therefore, it can't be used as an initial "
                          f"configuration as it does not match the current config space. ")
    return initial_configurations


def read_forecasting_init_configurations(config_space: ConfigurationSpace,
                                         suggested_init_models: Optional[List[str]] = None,
                                         custom_init_setting_path: Optional[str] = None,
                                         dataset_properties: Dict = {}
                                         ) -> List[Configuration]:
    forecasting_init_path = os.path.join(os.path.dirname(__file__), '../configs/forecasting_init_cfgs.json')
    initial_configurations_dict: List[Dict] = list()
    initial_configurations = []
    uni_variant = dataset_properties.get('uni_variant', True)
    targets_have_missing_values = dataset_properties.get('targets_have_missing_values', False)
    features_have_missing_values = dataset_properties.get('features_have_missing_values', False)

    if suggested_init_models or suggested_init_models is None:
        with open(forecasting_init_path, 'r') as f:
            forecasting_init_dict: Dict[str, Any] = json.load(f)
        cfg_trainer: Dict = forecasting_init_dict['trainer']
        models_name_to_cfgs: Dict = forecasting_init_dict['models']

        window_size = config_space.get_default_configuration()["data_loader:window_size"]
        if suggested_init_models is None:
            suggested_init_models = list(models_name_to_cfgs.keys())

        for model_name in suggested_init_models:
            cfg_tmp = cfg_trainer.copy()

            model_cfg = models_name_to_cfgs.get(model_name, None)
            if model_cfg is None:
                warnings.warn(f'Cannot to find the corresponding information of model {model_name} from,'
                              f' forecasting_init_cfgs, currently only {list(models_name_to_cfgs.keys())} are '
                              f'supported')
                continue
            if not model_cfg.get('data_loader:backcast', False):
                cfg_tmp['data_loader:window_size'] = window_size

            cfg_tmp.update(model_cfg)
            if not uni_variant:
                cfg_tmp.update(forecasting_init_dict['feature_preprocessing'])
                if features_have_missing_values:
                    cfg_tmp.update(forecasting_init_dict['feature_imputer'])
            if targets_have_missing_values:
                cfg_tmp.update(forecasting_init_dict['target_imputer'])

            initial_configurations_dict.append(cfg_tmp)

    if custom_init_setting_path is not None:
        try:
            with open(custom_init_setting_path, 'r') as f:
                initial_configurations_custom_dict: Union[List[Dict[str, Any]], Dict] = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError("The path: {} provided for 'custome_setting_path' for "
                                    "the file containing the custom initial configurations "
                                    "does not exist. Please provide a valid path".format(custom_init_setting_path))
        if isinstance(initial_configurations_custom_dict, list):
            initial_configurations_dict.extend(initial_configurations_custom_dict)
        else:
            initial_configurations_dict.append(initial_configurations_custom_dict)

    for configuration_dict in initial_configurations_dict:
        try:
            configuration = Configuration(config_space, configuration_dict)
            initial_configurations.append(configuration)
        except Exception as e:
            warnings.warn(f"Failed to convert {configuration_dict} into"
                          f" a Configuration with error {e}. "
                          f"Therefore, it can't be used as an initial "
                          f"configuration as it does not match the current config space. ")
    return initial_configurations
