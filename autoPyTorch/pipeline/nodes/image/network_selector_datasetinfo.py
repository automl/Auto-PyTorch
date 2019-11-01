__author__ = "Michael Burkart"
__version__ = "0.0.1"
__license__ = "BSD"


from autoPyTorch.pipeline.nodes.network_selector import NetworkSelector
from autoPyTorch.components.networks.base_net import BaseNet

import torch.nn as nn
import ConfigSpace
import ConfigSpace.hyperparameters as CSH
from autoPyTorch.utils.configspace_wrapper import ConfigWrapper
from autoPyTorch.utils.config.config_option import ConfigOption
import torchvision.models as models

class NetworkSelectorDatasetInfo(NetworkSelector):
    def fit(self, hyperparameter_config, pipeline_config, dataset_info):
        config = ConfigWrapper(self.get_name(), hyperparameter_config)
        network_name = config['network']

        network_type = self.networks[network_name]
        network_config = ConfigWrapper(network_name, config)
        activation = self.final_activations[pipeline_config["final_activation"]]

        in_features = dataset_info.x_shape[1:]
        if len(in_features) == 1:
            # feature data - otherwise image data (channels, width, height)
            in_features = in_features[0]

        network = network_type( config=network_config, 
                                in_features=in_features, out_features=dataset_info.y_shape[1],
                                final_activation=activation)

        # self.logger.debug('NETWORK:\n' + str(network))
        return {'network': network}
