from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace import ConfigurationSpace

from torch import nn

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import BaseEstimator
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.\
    base_forecasting_encoder import BaseForecastingEncoder, EncoderProperties
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_encoder.flat_encoder.\
    MLPEncoder import TimeSeriesMLP
from autoPyTorch.pipeline.components.setup.network_backbone.utils import get_output_shape
from autoPyTorch.utils.common import FitRequirement


class NBEATSEncoder(BaseForecastingEncoder):
    """
    Encoder for NBEATS-like network. It flatten the input sequence to fit the requirement of MLP, the main part is
    implemented under decoder
    """
    _fixed_seq_length = True
    window_size = 1

    @staticmethod
    def encoder_properties() -> EncoderProperties:
        return EncoderProperties(fixed_input_seq_length=True)

    @staticmethod
    def allowed_decoders() -> List[str]:
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
        self.check_requirements(X, y)
        self.window_size = X["window_size"]

        # n-BEATS only requires targets as its input
        # TODO add support for multi-variant
        output_shape = X["dataset_properties"]['output_shape']

        self.encoder = self.build_encoder(
            input_shape=output_shape,
        )

        input_shape = (self.window_size, output_shape[-1])
        self.input_shape = input_shape

        has_hidden_states = self.encoder_properties().has_hidden_states
        self.encoder_output_shape = get_output_shape(self.encoder, input_shape, has_hidden_states)
        return self

    def n_encoder_output_feature(self) -> int:
        # This function should never be called!!!
        raise NotImplementedError

    def build_encoder(self,
                      input_shape: Tuple[int, ...]) -> nn.Module:
        preprocessor = TimeSeriesMLP(window_size=self.window_size)
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
