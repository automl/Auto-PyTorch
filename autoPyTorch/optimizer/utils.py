import json
import os
import warnings
from typing import Any, Dict, List

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
