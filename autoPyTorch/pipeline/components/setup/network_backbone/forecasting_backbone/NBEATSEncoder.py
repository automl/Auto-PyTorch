from typing import Any, Dict, List, Optional, Union, Tuple

from torch import nn

from ConfigSpace import ConfigurationSpace

from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.base_forecasting_encoder import (
    BaseForecastingEncoder, EncoderNetwork
)
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.MLPEncoder import \
    TimeSeriesMLPrecpocessor


class NBEATSEncoder(BaseForecastingEncoder):
    """
    Encoder for NBEATS-like network. It flatten the input sequence to fit the requirement of MLP, the main part is
    implemented under decoder
    """
    _fixed_seq_length = True
    window_size = 1

    def encoder_properties(self):
        encoder_properties = super().encoder_properties()
        encoder_properties.update({
            'fixed_input_seq_length': True,
        })
        return encoder_properties

    @staticmethod
    def allowed_decoders():
        """
        decoder that is compatible with the encoder
        """
        return ['NBEATSDecoder']

    @property
    def _required_fit_arguments(self) -> List[FitRequirement]:
        requirements_list = super()._required_fit_arguments
        requirements_list.append(FitRequirement('window_size', (int,), user_defined=False, dataset_property=False))
        return requirements_list

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        self.window_size = X["window_size"]
        return super().fit(X, y)

    def build_encoder(self, input_shape: Tuple[int, ...]) -> nn.Module:
        preprocessor = TimeSeriesMLPrecpocessor(window_size=self.window_size)
        return preprocessor

    @staticmethod
    def get_properties(dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None
                       ) -> Dict[str, Union[str, bool]]:
        return {
            'shortname': 'NBEATSEncoder',
            'name': 'NBEATSEncoder',
            'handles_tabular': False,
            'handles_image': False,
            'handles_time_series': True,
        }

    @staticmethod
    def get_hyperparameter_search_space(
            dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        return cs
