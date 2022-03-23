import json
from math import floor
import os
import warnings
from typing import Any, Dict, List

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from autoPyTorch.utils.common import replace_string_bool_to_bool

def read_return_initial_configurations(
    config_space: ConfigurationSpace,
    portfolio_selection: str,
    num_numerical_features: int
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
            configuration_dict = validate_config(configuration_dict, config_space, num_numerical_features)
            configuration = Configuration(config_space, configuration_dict)
            initial_configurations.append(configuration)
        except Exception as e:
            warnings.warn(f"Failed to convert {configuration_dict} into"
                          f" a Configuration with error {e}. "
                          f"Therefore, it can't be used as an initial "
                          f"configuration as it does not match the current config space. ")
    return initial_configurations


def validate_config(config, search_space, num_numerical):
    modified_config = config.copy()
    for key, choice in config.items():
        if '__choice__' in key:
            choice_hyperparameter = search_space.get_hyperparameter(key)
            if choice not in choice_hyperparameter.choices:
                modified_config[key] = choice_hyperparameter.default_value
                dependent_hyperparamerters = {key: search_space.get_hyperparameter(key).default_value for key in config if choice_hyperparameter.default_value in key}
                modified_config = {key: modified_config[key] for key in modified_config if choice not in key}
                modified_config.update(dependent_hyperparamerters)

    config_has_imputer = any('imputer' in key for key in modified_config)
    search_space_has_imputer = 'imputer:numerical_strategy' in search_space.get_hyperparameter_names()
    if not config_has_imputer and search_space_has_imputer:
        modified_config['imputer:numerical_strategy'] = search_space.get_hyperparameter('imputer:numerical_strategy').default_value
    if config_has_imputer and not search_space_has_imputer:
        del modified_config['imputer:numerical_strategy']

    has_embedding = modified_config['network_embedding:__choice__'] == 'LearnedEntityEmbedding'

    if has_embedding:
        modified_config = {key: modified_config[key] for key in modified_config if 'LearnedEntityEmbedding' not in key}
        child_hps = search_space.get_children_of('network_embedding:__choice__')
        for hp in child_hps:
            modified_config[hp.name] = hp.default_value

    has_truncated_svd = modified_config['feature_preprocessor:__choice__'] == 'TruncatedSVD'
    if has_truncated_svd:
        target_dim = modified_config['feature_preprocessor:TruncatedSVD:target_dim']
        updated_target_dim = search_space.get_hyperparameter('feature_preprocessor:TruncatedSVD:target_dim').default_value if target_dim == 0 else max(floor(target_dim * num_numerical), 1)
        modified_config['feature_preprocessor:TruncatedSVD:target_dim'] = updated_target_dim

    return replace_string_bool_to_bool(modified_config)