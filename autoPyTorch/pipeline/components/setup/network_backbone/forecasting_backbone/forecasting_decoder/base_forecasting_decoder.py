from abc import abstractmethod
from collections import OrderedDict
from typing import Any, Dict, Iterable, List, Optional, Tuple

from torch import nn

from autoPyTorch.pipeline.components.base_component import (
    BaseEstimator, autoPyTorchComponent)
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.components_util import \
    NetworkStructure
from autoPyTorch.pipeline.components.setup.network_backbone.forecasting_backbone.forecasting_decoder.components import (
    DecoderBlockInfo, DecoderProperties)
from autoPyTorch.utils.common import FitRequirement


class BaseForecastingDecoder(autoPyTorchComponent):
    """
    Base class for network decoder used for forecasting. Holds the decoder module and the config which was used to
    create it.
    """
    _required_properties = ["name", "shortname", "handles_tabular", "handles_image", "handles_time_series"]

    def __init__(self,
                 block_number: int = 1,
                 auto_regressive: bool = False,
                 **kwargs: Any):
        super().__init__()
        self.block_number = block_number
        self.add_fit_requirements([
            FitRequirement('known_future_features', (tuple,), user_defined=False, dataset_property=True),
            FitRequirement('feature_shapes', (Dict,), user_defined=False, dataset_property=True),
            FitRequirement('network_encoder', (OrderedDict,), user_defined=False, dataset_property=False),
            FitRequirement('n_prediction_steps', (int,), user_defined=False, dataset_property=True),
            FitRequirement('network_structure', (NetworkStructure,), user_defined=False, dataset_property=False),
            FitRequirement('transform_time_features', (bool,), user_defined=False, dataset_property=False),
            FitRequirement('time_feature_transform', (Iterable,), user_defined=False, dataset_property=True)
        ])
        self.auto_regressive = auto_regressive
        self.config = kwargs
        self.decoder: Optional[nn.Module] = None
        self.n_decoder_output_features: Optional[int] = None
        self.decoder_input_shape: Optional[Tuple[int, ...]] = None
        self.n_prediction_heads = 1
        self.is_last_decoder: Optional[bool] = False

    @property
    def fitted_encoder(self) -> List[str]:
        return []

    @staticmethod
    def decoder_properties() -> DecoderProperties:
        return DecoderProperties()

    def fit(self, X: Dict[str, Any], y: Any = None) -> BaseEstimator:
        """
        Builds the head component and assigns it to self.decoder

        Args:
            X (X: Dict[str, Any]):
                Dependencies needed by current component to perform fit
            y (Any):
                not used. To comply with sklearn API
        Returns:
            Self
        """
        self.check_requirements(X, y)
        output_shape = X['dataset_properties']['output_shape']

        encoder_output_shape = X['network_encoder'][f'block_{self.block_number}'].encoder_output_shape

        auto_regressive = self.auto_regressive

        if auto_regressive:
            self.n_prediction_heads = 1
        else:
            self.n_prediction_heads = X['dataset_properties']['n_prediction_steps']

        network_structure = X['network_structure']
        variable_selection = network_structure.variable_selection

        if 'n_decoder_output_features' not in X:
            future_features = X['dataset_properties']['known_future_features']
            feature_shapes = X['dataset_properties']['feature_shapes']

            future_in_features = sum([feature_shapes[fu_feat] for fu_feat in future_features])

            if X['transform_time_features']:
                n_time_feature_transform = len(X['dataset_properties']['time_feature_transform'])
            else:
                n_time_feature_transform = 0

            if variable_selection:
                future_in_features = X['network_encoder']['block_1'].encoder_output_shape[-1]
            else:
                if auto_regressive:
                    if self.decoder_properties().lagged_input and hasattr(self, 'lagged_value'):
                        future_in_features += len(self.lagged_value) * output_shape[-1]
                    elif self.decoder_properties().recurrent:
                        future_in_features += output_shape[-1]
                future_in_features += n_time_feature_transform
            future_variable_input = (self.n_prediction_heads, future_in_features)
        else:
            future_variable_input = (self.n_prediction_heads, X['n_decoder_output_features'])

        if self.block_number == network_structure.num_blocks:
            self.is_last_decoder = True

        # TODO consider decoder auto regressive and fill in decoder part

        self.decoder, self.n_decoder_output_features = self.build_decoder(
            encoder_output_shape=encoder_output_shape,
            future_variable_input=future_variable_input,
            n_prediction_heads=self.n_prediction_heads,
            dataset_properties=X['dataset_properties']
        )

        self.decoder_input_shape = future_variable_input

        return self

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the network head into the fit dictionary 'X' and returns it.

        Args:
            X (Dict[str, Any]):
                'X' dictionary
        Returns:
            (Dict[str, Any]):
                the updated 'X' dictionary
        """
        # 'auto_regressive' needs to be the same across all the decoders,
        # 'n_prediction_heads' and 'n_decoder_output_features' are only applied to the head such that they could be
        # overwritten by the following decoders
        network_decoder = X.get('network_decoder', OrderedDict())
        assert self.decoder_input_shape is not None
        assert self.n_decoder_output_features is not None
        network_decoder[f'block_{self.block_number}'] = DecoderBlockInfo(
            decoder=self.decoder,
            decoder_properties=self.decoder_properties(),
            decoder_input_shape=self.decoder_input_shape,
            decoder_output_shape=(self.n_prediction_heads, self.n_decoder_output_features)
        )
        if self.is_last_decoder:
            X.update({'network_decoder': network_decoder,
                      'n_prediction_heads': self.n_prediction_heads,
                      'n_decoder_output_features': self.n_decoder_output_features,
                      'auto_regressive': self.auto_regressive})
        else:
            X.update({'network_decoder': network_decoder,
                      'n_decoder_output_features': self.n_decoder_output_features,
                      })

        return X

    def build_decoder(self,
                      encoder_output_shape: Tuple[int, ...],
                      future_variable_input: Tuple[int, ...],
                      n_prediction_heads: int,
                      dataset_properties: Dict) -> Tuple[nn.Module, int]:
        """
        Builds the head module and returns it

        Args:
            encoder_output_shape (Tuple[int, ...]):
                shape of the input to the decoder, this value is the encoder output
            future_variable_input (Tuple[int, ...]):
                shape of the known future input values
            n_prediction_heads (int):
                how many prediction heads the network has, used for final forecasting heads
            dataset_properties (Dict):
                dataset properties
        Returns:
            nn.Module:
                head module
        """
        decoder, n_decoder_features = self._build_decoder(encoder_output_shape, future_variable_input,
                                                          n_prediction_heads, dataset_properties)
        return decoder, int(n_decoder_features)

    @abstractmethod
    def _build_decoder(self,
                       encoder_output_shape: Tuple[int, ...],
                       future_variable_input: Tuple[int, ...],
                       n_prediction_heads: int,
                       dataset_properties: Dict) -> Tuple[nn.Module, int]:
        """
        Builds the head module and returns it

        Args:
            encoder_output_shape (Tuple[int, ...]):
                shape of the input to the decoder, this value is the encoder output
            future_variable_input (Tuple[int, ...]):
                shape of the known future input values
            n_prediction_heads (int):
                how many prediction heads the network has, used for final forecasting heads
            dataset_properties (Dict):
                dataset properties

        Returns:
            decoder (nn.Module):
                decoder module
            n_decoder_features (int):
                output of decoder features, used for initialize network head.
        """
        raise NotImplementedError()

    @classmethod
    def get_name(cls) -> str:
        """
        Get the name of the decoder

        Args:
            None

        Returns:
            str:
                Name of the decoder
        """
        return str(cls.get_properties()["shortname"])
