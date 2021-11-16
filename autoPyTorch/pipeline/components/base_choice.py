import re
import warnings
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

import numpy as np

from sklearn.utils import check_random_state

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.utils.common import FitRequirement, HyperparameterSearchSpace
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdate


class autoPyTorchChoice(object):
    """
    Allows for the dynamically generation of components as pipeline steps.

    Args:
        dataset_properties (Dict[str, Union[str, BaseDatasetPropertiesType]]):
            Describes the dataset to work on
        random_state (Optional[np.random.RandomState]):
            Allows to produce reproducible results by setting a
            seed for randomized settings

    Attributes:
        random_state (Optional[np.random.RandomState]):
            Allows to produce reproducible results by setting a seed for
            randomized settings
        choice (autoPyTorchComponent):
            the choice of components for this stage
    """
    def __init__(self,
                 dataset_properties: Dict[str, BaseDatasetPropertiesType],
                 random_state: Optional[np.random.RandomState] = None
                 ):

        # Since all calls to get_hyperparameter_search_space will be done by the
        # pipeline on construction, it is not necessary to construct a
        # configuration space at this location!
        # self.configuration = self.get_hyperparameter_search_space(
        #     dataset_properties).get_default_configuration()

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        self.dataset_properties = dataset_properties
        self._check_dataset_properties(dataset_properties)
        # Since the pipeline will initialize the hyperparameters, it is not
        # necessary to do this upon the construction of this object
        # self.set_hyperparameters(self.configuration)
        self.choice: Optional[autoPyTorchComponent] = None

        self._cs_updates: Dict[str, HyperparameterSearchSpaceUpdate] = dict()

    def get_fit_requirements(self) -> Optional[List[FitRequirement]]:
        if self.choice is not None:
            return self.choice.get_fit_requirements()
        else:
            raise AttributeError("Expected choice attribute to be autoPyTorchComponent"
                                 " but got None, to get fit requirements for {}, "
                                 "call get_fit_requirements of the component".format(self.__class__.__name__))

    def get_components(cls: 'autoPyTorchChoice') -> Dict[str, autoPyTorchComponent]:
        """Returns and ordered dict with the components available
        for current step.

        Args:
            cls (autoPyTorchChoice):
                The choice object from which to query the valid
                components

        Returns:
            Dict[str, autoPyTorchComponent]:
                The available components via a mapping
                from the module name to the component class

        """
        raise NotImplementedError()

    def get_available_components(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> Dict[str, autoPyTorchComponent]:
        """
        Wrapper over get components to incorporate include/exclude
        user specification

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
                Describes the dataset to work on
            include: Optional[Dict[str, Any]]:
                what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]:
                which components to skip. Can't be used together with include

        Results:
            Dict[str, autoPyTorchComponent]: A dictionary with valid components for this
                choice object

        """
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue
            if 'issparse' in dataset_properties:
                if dataset_properties['issparse'] and \
                        not available_comp[name].get_properties(dataset_properties)['handles_sparse']:
                    continue
            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(self,
                            configuration: Configuration,
                            init_params: Optional[Dict[str, Any]] = None
                            ) -> 'autoPyTorchChoice':
        """
        Applies a configuration to the given component.
        This method translate a hierarchical configuration key,
        to an actual parameter of the autoPyTorch component.

        Args:
            configuration (Configuration):
                Which configuration to apply to the chosen component
            init_params (Optional[Dict[str, any]]):
                Optional arguments to initialize the chosen component

        Returns:
            self: returns an instance of self
        """
        new_params = {}

        params = configuration.get_dictionary()
        choice = params['__choice__']
        del params['__choice__']

        for param, value in params.items():
            param = param.replace(choice + ':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice + ':', '')
                new_params[param] = value

        new_params['random_state'] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def get_hyperparameter_search_space(
        self,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        default: Optional[str] = None,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> ConfigurationSpace:
        """Returns the configuration space of the current chosen components

        Args:
            dataset_properties (Optional[Dict[str, BaseDatasetPropertiesType]]):
                Describes the dataset to work on
            default: (Optional[str]):
                Default component to use in hyperparameters
            include: Optional[Dict[str, Any]]:
                what components to include. It is an exhaustive
                list, and will exclusively use this components.
            exclude: Optional[Dict[str, Any]]:
                which components to skip

        Returns:
            ConfigurationSpace: the configuration space of the hyper-parameters of the
                chosen component
        """
        raise NotImplementedError()

    def fit(self, X: Dict[str, Any], y: Any) -> autoPyTorchComponent:
        """Handy method to check if a component is fitted

        Args:
            X (X: Dict[str, Any]):
                Dependencies needed by current component to perform fit
            y (Any):
                not used. To comply with sklearn API
        """
        # Allows to use check_is_fitted on the choice object
        self.fitted_ = True
        assert self.choice is not None, "Cannot call fit without initializing the component"
        return self.choice.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predicts the target given an input, by using the chosen component

        Args:
            X (np.ndarray):
                input features from which to predict the target

        Returns:
            np.ndarray:
                the target prediction
        """
        assert self.choice is not None, "Cannot call predict without initializing the component"
        return self.choice.predict(X)

    def transform(self, X: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adds the current choice in the fit dictionary

        Args:
            X (Dict[str, Any]):
                fit dictionary

        Returns:
            (Dict[str, Any])
        """
        assert self.choice is not None, "Can not call transform without initialising the component"
        X = self.choice.transform(X)
        return X

    def check_requirements(self, X: Dict[str, Any], y: Any = None) -> None:
        """
        A mechanism in code to ensure the correctness of the fit dictionary
        It recursively makes sure that the children and parent level requirements
        are honored before fit.

        Args:
            X (Dict[str, Any]):
                Dictionary with fitted parameters. It is a message passing
                mechanism, in which during a transform, a components adds relevant information
                so that further stages can be properly fitted
        """
        assert isinstance(X, dict), "The input X to the pipeline must be a dictionary"

        if y is not None:
            warnings.warn("Provided y argument, yet only X is required")

    def _check_dataset_properties(self, dataset_properties: Dict[str, BaseDatasetPropertiesType]) -> None:
        """
        A mechanism in code to ensure the correctness of the initialised dataset properties.
        Args:
            dataset_properties (Dict[str, BaseDatasetPropertiesType]):
                Describes the dataset to work on

        """
        assert isinstance(dataset_properties, dict), "dataset_properties must be a dictionary"

    def _apply_search_space_update(self, hyperparameter_search_space_update: HyperparameterSearchSpaceUpdate) -> None:
        """
        Applies search space update to the class

        Args:
            hyperparameter_search_space_update (HyperparameterSearchSpaceUpdate):
                Search Space update for the current autoPyTorchChoice module
        """

        self._cs_updates[hyperparameter_search_space_update.hyperparameter] = hyperparameter_search_space_update

    def _get_search_space_updates(self, prefix: Optional[str] = None) -> Dict[str, HyperparameterSearchSpace]:
        """Get the search space updates with the given prefix

        Args:
            prefix (str):
                Only return search space updates with given prefix (default: {None})

        Returns:
            Dict[str, HyperparameterSearchSpace]:
                Mapping of search space updates. Keys don't contain the prefix.
        """

        result: Dict[str, HyperparameterSearchSpace] = dict()

        # iterate over all search space updates of this node and keep the ones that have the given prefix
        for key in self._cs_updates.keys():
            if prefix is None:
                result[key] = self._cs_updates[key].get_search_space()
            elif re.search(f'^{prefix}', key) is not None:
                result[key[len(prefix) + 1:]] = self._cs_updates[key].get_search_space(remove_prefix=prefix)
        return result
