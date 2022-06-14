import warnings
from abc import ABCMeta
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state

import torch

from autoPyTorch.datasets.base_dataset import BaseDatasetPropertiesType
from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.create_searchspace_util import (
    add_forbidden,
    find_active_choices,
    get_match_array
)
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import HyperparameterSearchSpaceUpdates


PipelineStepType = Union[autoPyTorchComponent, autoPyTorchChoice]


class BasePipeline(Pipeline):
    """
    Base class for all pipeline objects.

    Args:
        config (Optional[Configuration]):
            Allows to directly specify a configuration space
        steps (Optional[List[Tuple[str, PipelineStepType]]]):
            The list of `autoPyTorchComponent` or `autoPyTorchChoice`
            that build the pipeline. If provided, they won't be
            dynamically produced.
        include (Optional[Dict[str, Any]]):
            Allows the caller to specify which configurations to honor during
            the creation of the configuration space.
        exclude (Optional[Dict[str, Any]]):
            Allows the caller to specify which configurations
            to avoid during the creation of the configuration space.
        random_state (np.random.RandomState):
            allows to produce reproducible results by
            setting a seed for randomized settings
        init_params (Optional[Dict[str, Any]]):
            Optional initial settings for the config
        search_space_updates (Optional[HyperparameterSearchSpaceUpdates]):
            search space updates that can be used to modify the search
            space of particular components or choice modules of the pipeline

    Attributes:
        steps (List[Tuple[str, PipelineStepType]]):
            the steps of the current pipeline. Each step in an AutoPyTorch
            pipeline is either a autoPyTorchChoice or autoPyTorchComponent.
            Both of these are child classes of sklearn 'BaseEstimator' and
            they perform operations on and transform the fit dictionary.
            For more info, check documentation of 'autoPyTorchChoice' or
            'autoPyTorchComponent'.
        config (Configuration):
            a configuration to delimit the current component choice
        random_state (Optional[np.random.RandomState]):
            allows to produce reproducible
               results by setting a seed for randomized settings

    Notes:
        This class should not be instantiated, only subclassed.
    """
    __metaclass__ = ABCMeta

    def __init__(
        self,
        config: Optional[Configuration] = None,
        steps: Optional[List[Tuple[str, PipelineStepType]]] = None,
        dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]] = None,
        include: Optional[Dict[str, Any]] = None,
        exclude: Optional[Dict[str, Any]] = None,
        random_state: Optional[np.random.RandomState] = None,
        init_params: Optional[Dict[str, Any]] = None,
        search_space_updates: Optional[HyperparameterSearchSpaceUpdates] = None
    ):

        self.init_params = init_params if init_params is not None else {}
        self.dataset_properties = dataset_properties if \
            dataset_properties is not None else {}
        self.include = include if include is not None else {}
        self.exclude = exclude if exclude is not None else {}
        self.search_space_updates = search_space_updates
        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        if steps is None:
            self.steps = self._get_pipeline_steps(dataset_properties)
        else:
            self.steps = steps

        self.config_space = self.get_hyperparameter_search_space()

        if config is None:
            self.config = self.config_space.get_default_configuration()
        else:
            if isinstance(config, dict):
                config = Configuration(self.config_space, config)
            if self.config_space != config.configuration_space:
                warnings.warn(self.config_space._children)
                warnings.warn(config.configuration_space._children)
                import difflib
                diff = difflib.unified_diff(
                    str(self.config_space).splitlines(),
                    str(config.configuration_space).splitlines())
                diff_msg = '\n'.join(diff)
                raise ValueError('Configuration passed does not come from the '
                                 'same configuration space. Differences are: '
                                 '%s' % diff_msg)
            self.config = config

        self.set_hyperparameters(self.config, init_params=init_params)

        super().__init__(steps=self.steps)

        self._additional_run_info: Dict[str, str] = {}

    def fit(self, X: Dict[str, Any], y: Optional[np.ndarray] = None,
            **fit_params: Any) -> Pipeline:
        """Fit the selected algorithm to the training data.

        Args:
            X (typing.Dict):
            A fit dictionary that contains information to fit a pipeline
            TODO: Use fit_params support from 0.24 scikit learn version instead
            y (None):
            Used for Compatibility, but it has no funciton in out fit strategy
            TODO: use actual y when moving to fit_params support
        fit_params : dict
            See the documentation of sklearn.pipeline.Pipeline for formatting
            instructions.

        Returns:
            self :
                returns an instance of self.

        Raises:
            NoModelException
                NoModelException is raised if fit() is called without specifying
                a classification algorithm first.
        """
        X, fit_params = self.fit_transformer(X, y, **fit_params)
        self.fit_estimator(X, y, **fit_params)
        return self

    def fit_transformer(self, X: Dict[str, Any], y: Optional[np.ndarray] = None,
                        fit_params: Optional[Dict] = None,
                        ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if fit_params is None:
            fit_params = {}
        fit_params = {key.replace(":", "__"): value for key, value in
                      fit_params.items()}
        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, **fit_params_steps)
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit_estimator(self, X: Dict[str, Any],
                      y: Optional[np.ndarray], **fit_params: Any
                      ) -> Pipeline:
        fit_params = {key.replace(":", "__"): value for key, value in
                      fit_params.items()}
        self._final_estimator.fit(X, y, **fit_params)
        return self

    def predict(self, X: np.ndarray, batch_size: Optional[int] = None) -> np.ndarray:
        """Predict the output using the selected model.

        Args:
            X (np.ndarray): input data to the array
            batch_size (Optional[int]): batch_size controls whether the pipeline will be
                called on small chunks of the data. Useful when calling the
                predict method on the whole array X results in a MemoryError.

        Returns:
            np.ndarray: the predicted values given input X
        """

        # Pre-process X
        if batch_size is None:
            warnings.warn("Batch size not provided. "
                          "Will predict on the whole data in a single iteration")
            batch_size = X.shape[0]
        loader = self.named_steps['data_loader'].get_loader(X=X, batch_size=batch_size)
        return self.named_steps['network'].predict(loader)

    def set_hyperparameters(
            self,
            configuration: Configuration,
            init_params: Optional[Dict] = None
    ) -> 'Pipeline':
        """Method to set the hyperparameter configuration of the pipeline.

        It iterates over the components of the pipeline and applies a given
        configuration accordingly.

        Args:
            configuration (Configuration): configuration object to search and overwrite in
                the pertinent spaces
            init_params (Optional[Dict]): optional initial settings for the config

        """
        self.configuration = configuration

        for node_idx, n_ in enumerate(self.steps):
            node_name, node = n_

            updates: Dict[str, Any] = {}
            if not isinstance(node, autoPyTorchChoice):
                updates = node._get_search_space_updates()

            sub_configuration_space = node.get_hyperparameter_search_space(  # type: ignore[call-arg]
                self.dataset_properties,
                **updates
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % node_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % node_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space,
                                              values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % node_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % node_name, '', 1)
                        sub_init_params_dict[new_name] = value

            if isinstance(node, (autoPyTorchChoice, autoPyTorchComponent, BasePipeline)):
                node.set_hyperparameters(
                    configuration=sub_configuration,
                    init_params=None if init_params is None else sub_init_params_dict,
                )
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.

        Returns:
            ConfigurationSpace: The configuration space describing the Pipeline.
        """
        if not hasattr(self, 'config_space') or self.config_space is None:
            self.config_space = self._get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties,
                include=self.include,
                exclude=self.exclude,
            )
        return self.config_space

    def get_model(self) -> torch.nn.Module:
        """
        Returns the fitted model to the user
        """
        return self.named_steps['network'].get_network()

    def _get_hyperparameter_search_space(self,
                                         dataset_properties: Dict[str, BaseDatasetPropertiesType],
                                         include: Optional[Dict[str, Any]] = None,
                                         exclude: Optional[Dict[str, Any]] = None,
                                         ) -> ConfigurationSpace:
        """Return the configuration space for the CASH problem.
        This method should be called by the method
        get_hyperparameter_search_space of a subclass. After the subclass
        assembles a list of available estimators and preprocessor components,
        _get_hyperparameter_search_space can be called to do the work of
        creating the actual ConfigSpace.configuration_space.ConfigurationSpace object.

        Args:
            include (Dict): Overwrite to include user desired components to the pipeline
            exclude (Dict): Overwrite to exclude user desired components to the pipeline

        Returns:
            Configuration: The configuration space describing the AutoPytorch estimator.
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        """Retrieves a str representation of the current pipeline

        Returns:
            str: A formatted representation of the pipeline stages
                 and components
        """
        string = ''
        string += '_' * 40
        string += "\n\t" + self.__class__.__name__ + "\n"
        string += '_' * 40
        string += "\n"
        for i, (stage_name, component) in enumerate(self.named_steps.items()):
            string += str(i) + "-) " + stage_name + ": "
            string += "\n\t"
            string += str(component.choice) if hasattr(component, 'choice') else str(component)
            string += "\n"
            string += "\n"
        string += '_' * 40
        return string

    def _get_base_search_space(
            self,
            cs: ConfigurationSpace,
            dataset_properties: Dict[str, BaseDatasetPropertiesType],
            include: Optional[Dict[str, Any]],
            exclude: Optional[Dict[str, Any]],
            pipeline: List[Tuple[str, PipelineStepType]]
    ) -> ConfigurationSpace:
        if include is None:
            include = self.include

        keys = [pair[0] for pair in pipeline]
        for key in include:
            if key not in keys:
                raise ValueError('Invalid key in include: %s; should be one '
                                 'of %s' % (key, keys))

        if exclude is None:
            exclude = self.exclude

        keys = [pair[0] for pair in pipeline]
        for key in exclude:
            if key not in keys:
                raise ValueError('Invalid key in exclude: %s; should be one '
                                 'of %s' % (key, keys))

        if self.search_space_updates is not None:
            self._check_search_space_updates(include=include,
                                             exclude=exclude)
            self.search_space_updates.apply(pipeline=pipeline)

        matches = get_match_array(
            pipeline, dataset_properties, include=include, exclude=exclude)

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        assert np.sum(matches) != 0, "No valid pipeline found."

        assert np.sum(matches) <= np.size(matches), \
            "'matches' is not binary; %s <= %d, %s" % \
            (str(np.sum(matches)), np.size(matches), str(matches.shape))

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, n_ in enumerate(pipeline):
            node_name, node = n_

            # if the node isn't a choice we can add it immediately because it
            #  must be active (if it wasn't, np.sum(matches) would be zero
            if isinstance(node, autoPyTorchChoice):
                choices_list = find_active_choices(
                    matches, node, node_idx,
                    dataset_properties,
                    include.get(node_name),
                    exclude.get(node_name)
                )

                # ignore type check here as mypy is not able to infer
                # that isinstance(node, autoPyTorchChooice) = True
                sub_config_space = node.get_hyperparameter_search_space(  # type: ignore[call-arg]
                    dataset_properties, include=choices_list)
                cs.add_configuration_space(node_name, sub_config_space)

            # If the node is a choice, we have to figure out which of its
            #  choices are actually legal choices
            else:
                cs.add_configuration_space(
                    node_name,
                    node.get_hyperparameter_search_space(dataset_properties,  # type: ignore[call-arg]
                                                         **node._get_search_space_updates()
                                                         )
                )

        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = add_forbidden(
                conf_space=cs, pipeline=pipeline, matches=matches,
                dataset_properties=dataset_properties, include=include,
                exclude=exclude)

        return cs

    def _check_search_space_updates(self, include: Optional[Dict[str, Any]],
                                    exclude: Optional[Dict[str, Any]]) -> None:
        assert self.search_space_updates is not None
        for update in self.search_space_updates.updates:
            if update.node_name not in self.named_steps.keys():
                raise ValueError("Unknown node name. Expected update node name to be in {} "
                                 "got {}".format(self.named_steps.keys(), update.node_name))
            node = self.named_steps[update.node_name]
            # if node is a choice module
            if hasattr(node, 'get_components'):
                split_hyperparameter = update.hyperparameter.split(':')

                # check if component is not present in include
                if include is not None and update.node_name in include.keys():
                    if split_hyperparameter[0] not in include[update.node_name]:
                        hp_in_component = False
                        # If the node contains subcomponent that is also an instance of autoPyTorchChoice,
                        # We need to ensure that include is properly passed to it subcomponent
                        for include_component in include[update.node_name]:
                            if include_component.startswith(split_hyperparameter[0]):
                                hp_in_component = True
                                break
                        if not hp_in_component:
                            raise ValueError("Not found {} in include".format(split_hyperparameter[0]))

                # check if component is present in exclude
                if exclude is not None and update.node_name in exclude.keys():
                    if split_hyperparameter[0] in exclude[update.node_name]:
                        hp_in_component = False
                        for exclude_component in exclude[update.node_name]:
                            if exclude_component.startswith(split_hyperparameter[0]):
                                hp_in_component = True
                                break
                        if not hp_in_component:
                            raise ValueError("Found {} in exclude".format(split_hyperparameter[0]))

                components = node.get_components()
                # if hyperparameter is __choice__, check if
                # the components in the value range of search space update
                # are in components of the choice module
                if split_hyperparameter[0] == '__choice__':
                    for choice in update.value_range:
                        if include is not None and update.node_name in include.keys():
                            if choice not in include[update.node_name]:
                                raise ValueError("Not found {} in include".format(choice))
                        if exclude is not None and update.node_name in exclude.keys():
                            if choice in exclude[update.node_name]:
                                raise ValueError("Found {} in exclude".format(choice))
                        if choice not in components.keys():
                            raise ValueError("Unknown hyperparameter for choice {}. "
                                             "Expected update hyperparameter "
                                             "to be in {} got {}".format(node.__class__.__name__,
                                                                         components.keys(), choice))
                # check if the component whose hyperparameter
                # needs to be updated is in components of the
                # choice module
                elif split_hyperparameter[0] not in components.keys():
                    hp_in_component = False
                    if hasattr(node, 'additional_components') and node.additional_components:
                        # This is designed for forecasting network encoder:
                        # forecasting network backbone is composed of two parts: encoder and decoder whereas the type
                        # of the decoder is determined by the encoder. However, the type of decoder cannot be any part
                        # of encoder's choice. To allow the user to update the hyperparameter search space for decoder
                        # network, we consider decoder as "additional_components" and check if the update can be applied
                        # to node.additional_components
                        for component_func in node.additional_components:
                            if split_hyperparameter[0] in component_func().keys():
                                hp_in_component = True
                                break
                    if not hp_in_component:
                        raise ValueError("Unknown hyperparameter for choice {}. "
                                         "Expected update hyperparameter "
                                         "to be in {} got {}".format(node.__class__.__name__,
                                                                     components.keys(), split_hyperparameter[0]))
                else:
                    # check if hyperparameter is in the search space of the component
                    component = components[split_hyperparameter[0]]
                    if split_hyperparameter[1] not in component. \
                            get_hyperparameter_search_space(dataset_properties=self.dataset_properties):
                        # Check if update hyperparameter is in names of
                        # hyperparameters of the search space
                        # Example 'num_units' in 'num_units_1', 'num_units_2'
                        if any([split_hyperparameter[1] in name for name in
                                component.get_hyperparameter_search_space(
                                    dataset_properties=self.dataset_properties).get_hyperparameter_names()]):
                            continue
                        raise ValueError("Unknown hyperparameter for component {}. "
                                         "Expected update hyperparameter "
                                         "to be in {} got {}".format(node.__class__.__name__,
                                                                     component.
                                                                     get_hyperparameter_search_space(
                                                                         dataset_properties=self.dataset_properties).
                                                                     get_hyperparameter_names(),
                                                                     split_hyperparameter[1]))
            else:
                if update.hyperparameter not in node.get_hyperparameter_search_space(
                        dataset_properties=self.dataset_properties):
                    if any([update.hyperparameter in name for name in
                            node.get_hyperparameter_search_space(
                                dataset_properties=self.dataset_properties).get_hyperparameter_names()]):
                        continue
                    raise ValueError("Unknown hyperparameter for component {}. "
                                     "Expected update hyperparameter "
                                     "to be in {} got {}".format(node.__class__.__name__,
                                                                 node.
                                                                 get_hyperparameter_search_space(
                                                                     dataset_properties=self.dataset_properties).
                                                                 get_hyperparameter_names(), update.hyperparameter))

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, BaseDatasetPropertiesType]]
                            ) -> List[Tuple[str, PipelineStepType]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoices.

        Returns:
            List[Tuple[str, PipelineStepType]]: list of steps sequentially exercised
                by the pipeline.
        """
        raise NotImplementedError()

    def get_fit_requirements(self) -> List[FitRequirement]:
        """
        Utility function that goes through all the components in
        the pipeline and gets the fit requirement of that components.
        All the fit requirements are then aggregated into a list

        Returns:
            List[NamedTuple]: List of FitRequirements
        """
        fit_requirements = list()  # List[FitRequirement]
        for name, step in self.steps:
            step_requirements = step.get_fit_requirements()
            if step_requirements:
                fit_requirements.extend(step_requirements)

        # remove duplicates in the list
        fit_requirements = list(set(fit_requirements))
        fit_requirements = [req for req in fit_requirements if (req.user_defined and not req.dataset_property)]
        req_names = [req.name for req in fit_requirements]

        # check wether requirement names are unique
        if len(set(req_names)) != len(fit_requirements):
            name_occurences = Counter(req_names)
            multiple_names = [name for name, num_occ in name_occurences.items() if num_occ > 1]
            multiple_fit_requirements = [req for req in fit_requirements if req.name in multiple_names]
            raise ValueError("Found fit requirements with different values %s" % multiple_fit_requirements)
        return fit_requirements

    def get_dataset_requirements(self) -> List[FitRequirement]:
        """
        Utility function that goes through all the components in
        the pipeline and gets the fit requirement that are expected to be
        computed by the dataset for that components. All the fit requirements
        are then aggregated into a list.

        Returns:
            List[NamedTuple]: List of FitRequirements
        """
        fit_requirements: List[FitRequirement] = list()
        for name, step in self.steps:
            step_requirements = step.get_fit_requirements()
            if step_requirements:
                fit_requirements.extend(step_requirements)

        # remove duplicates in the list
        fit_requirements = list(set(fit_requirements))
        fit_requirements = [req for req in fit_requirements if (req.user_defined and req.dataset_property)]
        return fit_requirements

    def _get_estimator_hyperparameter_name(self) -> str:
        """The name of the current pipeline estimator, for representation purposes"""
        raise NotImplementedError()

    def get_additional_run_info(self) -> Dict:
        """Allows retrieving additional run information from the pipeline.
        Can be overridden by subclasses to return additional information to
        the optimization algorithm.

        Returns:
            Dict: Additional information about the pipeline
        """
        return self._additional_run_info

    def get_pipeline_representation(self) -> Dict[str, str]:
        """
        Returns a representation of the pipeline, so that it can be
        consumed and formatted by the API.

        It should be a representation that follows:
        [{'PreProcessing': <>, 'Estimator': <>}]

        Returns:
            Dict: contains the pipeline representation in a short format
        """
        raise NotImplementedError()

    @staticmethod
    def get_default_pipeline_options() -> Dict[str, Any]:
        return {
            'num_run': 0,
            'device': 'cpu',
            'budget_type': 'epochs',
            'epochs': 5,
            'runtime': 3600,
            'torch_num_threads': 1,
            'early_stopping': 10,
            'use_tensorboard_logger': True,
            'metrics_during_training': True
        }
