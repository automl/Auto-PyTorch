import warnings
from abc import ABCMeta
from collections import Counter
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_random_state

import torch

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent
from autoPyTorch.pipeline.create_searchspace_util import (
    add_forbidden,
    find_active_choices,
    get_match_array
)
from autoPyTorch.utils.common import FitRequirement
from autoPyTorch.utils.hyperparameter_search_space_update import (
    HyperparameterSearchSpaceUpdates
)


def _err_msg(var_name: str, choices: Iterable, val: Any) -> str:
    return "Expected {} to be in {}, but got {}".format(var_name, list(choices), val)


class BasePipeline(Pipeline):
    """Base class for all pipeline objects.
    Notes
    -----
    This class should not be instantiated, only subclassed.

    Args:
        config (Optional[Configuration]): Allows to directly specify a configuration space
        steps (Optional[List[Tuple[str, autoPyTorchChoice]]]): the list of steps that
            build the pipeline. If provided, they won't be dynamically produced.
        include (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to honor during the creation of the configuration space.
        exclude (Optional[Dict[str, Any]]): Allows the caller to specify which configurations
            to avoid during the creation of the configuration space.
        random_state (np.random.RandomState): allows to produce reproducible results by
            setting a seed for randomized settings
        init_params (Optional[Dict[str, Any]])


    Attributes:
        steps (List[Tuple[str, autoPyTorchChoice]]]): the steps of the current pipeline
        config (Configuration): a configuration to delimit the current component choice
        random_state (Optional[np.random.RandomState]): allows to produce reproducible
               results by setting a seed for randomized settings

    """
    __metaclass__ = ABCMeta

    def __init__(
            self,
            config: Optional[Configuration] = None,
            steps: Optional[List[Tuple[str, autoPyTorchChoice]]] = None,
            dataset_properties: Optional[Dict[str, Any]] = None,
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

        self._additional_run_info = {}  # type: Dict[str, str]

    def fit(self, X: Dict[str, Any], y: Optional[np.ndarray] = None,
            **fit_params: Any) -> Pipeline:
        """Fit the selected algorithm to the training data.
        Arguments:
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

    def get_max_iter(self) -> int:
        if self.estimator_supports_iterative_fit():
            return self._final_estimator.get_max_iter()
        else:
            raise NotImplementedError()

    def configuration_fully_fitted(self) -> bool:
        return self._final_estimator.configuration_fully_fitted()

    def get_current_iter(self) -> int:
        return self._final_estimator.get_current_iter()

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

            sub_configuration_space = node.get_hyperparameter_search_space(self.dataset_properties,
                                                                           **updates)
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
                                         dataset_properties: Dict[str, Any],
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

    def _get_search_space_modifications(
        self,
        include: Optional[Dict[str, Any]],
        exclude: Optional[Dict[str, Any]],
        pipeline: List[Tuple[str, autoPyTorchChoice]]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Get what pipeline or components to
        include in or exclude from the searching space

        Args:
            include (Optional[Dict[str, Any]]): Components to include in the searching
            exclude (Optional[Dict[str, Any]]): Components to exclude from the searching
            pipeline (List[Tuple[str, autoPyTorchChoice]]):
                Available components

        Returns:
            include, exclude (Tuple[Dict[str, Any]], Dict[str, Any]]]):
                modified version of `include` and `exclude`
        """

        key_exist = {pair[0]: True for pair in pipeline}

        if include is None:
            include = {} if self.include is None else self.include

        for key in include.keys():
            if key_exist.get(key, False):
                raise ValueError('Keys in `include` must be {}, but got {}'.format(key_exist.keys(), key))

        if exclude is None:
            exclude = {} if self.exclude is None else self.exclude

        for key in exclude:
            if key_exist.get(key, False):
                raise ValueError('Keys in `exclude` must be {}, but got {}'.format(key_exist.keys(), key))

        return include, exclude

    @staticmethod
    def _update_search_space_per_node(
        cs: ConfigurationSpace,
        dataset_properties: Dict[str, Any],
        include: Dict[str, Any],
        exclude: Dict[str, Any],
        matches: np.ndarray,
        node_name: str,
        node_idx: int,
        node: autoPyTorchChoice
    ) -> ConfigurationSpace:
        """
        Args:
            cs (ConfigurationSpace): Searching space information
            dataset_properties (Dict[str, Any]): The properties of dataset
            include (Dict[str, Any]): Components to include in the searching
            exclude (Dict[str, Any]): Components to exclude from the searching
            node_name (str): The name of the component choice
            node_idx (int): The index of the component in a provided list of components
            node (autoPyTorchChoice): The module of the component
            matches (np.ndarray): ...

        Returns:
            modified cs (ConfigurationSpace):
                modified searching space information based on the arguments.
        """
        is_choice = isinstance(node, autoPyTorchChoice)

        if not is_choice:
            # if the node isn't a choice we can add it immediately because
            # it must be active (if it wasn't, np.sum(matches) would be zero
            assert not isinstance(node, autoPyTorchChoice)
            cs.add_configuration_space(
                node_name,
                node.get_hyperparameter_search_space(dataset_properties,  # type: ignore[arg-type]
                                                     **node._get_search_space_updates()),
            )
        else:
            # If the node is a choice, we have to figure out which of
            # its choices are actually legal choices
            choices_list = find_active_choices(
                matches, node, node_idx,
                dataset_properties,
                include.get(node_name),
                exclude.get(node_name)
            )
            sub_config_space = node.get_hyperparameter_search_space(
                dataset_properties, include=choices_list)
            cs.add_configuration_space(node_name, sub_config_space)

        return cs

    def _get_base_search_space(
            self,
            cs: ConfigurationSpace,
            dataset_properties: Dict[str, Any],
            include: Optional[Dict[str, Any]],
            exclude: Optional[Dict[str, Any]],
            pipeline: List[Tuple[str, autoPyTorchChoice]]
    ) -> ConfigurationSpace:
        """
        Get the searching space

        Args:
            cs (ConfigurationSpace): Searching space information
            dataset_properties (Dict[str, Any]): The properties of dataset
            include (Optional[Dict[str, Any]]): Components to include in the searching
            exclude (Optional[Dict[str, Any]]): Components to exclude from the searching
            pipeline (List[Tuple[str, autoPyTorchChoice]]):
                Available components

        Returns:
            modified cs (ConfigurationSpace):
                modified searching space information based on the arguments.
        """
        include, exclude = self._get_search_space_modifications(
            include=include, exclude=exclude, pipeline=pipeline
        )
        if self.search_space_updates is not None:
            self._check_search_space_updates(include=include,
                                             exclude=exclude)
            self.search_space_updates.apply(pipeline=pipeline)

        # The size of this array exponentially grows, so it will be better to remove
        matches = get_match_array(
            pipeline, dataset_properties, include=include, exclude=exclude)

        # Now we have only legal combinations at this step of the pipeline
        # Simple sanity checks
        if np.sum(matches) == 0:
            raise ValueError("No valid pipeline found.")
        if np.sum(matches) > np.size(matches):
            raise TypeError("'matches' is not binary; {} <= {}, {}".format(
                np.sum(matches), np.size(matches), str(matches.shape)
            ))

        # Iterate each dimension of the matches array (each step of the
        # pipeline) to see if we can add a hyperparameter for that step
        for node_idx, (node_name, node) in enumerate(pipeline):
            cs = self._update_search_space_per_node(
                cs=ConfigurationSpace, dataset_properties=dataset_properties,
                include=include, exclude=exclude, matches=matches,
                node=node, node_idx=node_idx, node_name=node_name
            )

        # And now add forbidden parameter configurations
        # According to matches
        if np.sum(matches) < np.size(matches):
            cs = add_forbidden(
                conf_space=cs, pipeline=pipeline, matches=matches,
                dataset_properties=dataset_properties, include=include,
                exclude=exclude)

        return cs

    @staticmethod
    def _check_valid_component(
        update: HyperparameterSearchSpaceUpdates,
        module_name: str,
        include: Optional[Dict[str, Any]],
        exclude: Optional[Dict[str, Any]]
    ) -> None:

        exist_in_include = include is not None and include.get(update.node_name, False)
        if exist_in_include:
            raise ValueError("Not found {} in include".format(module_name))

        # check if component is present in exclude
        exist_in_exclude = exclude is not None and exclude.get(update.node_name, False)
        if exist_in_exclude:
            raise ValueError("Found {} in exclude".format(module_name))

    def _check_available_components(
        self,
        update: HyperparameterSearchSpaceUpdates,
        include: Optional[Dict[str, Any]],
        exclude: Optional[Dict[str, Any]]
    ) -> None:

        component_names = update.hyperparameter.split(':')
        node = self.named_steps[update.node_name]
        node_name = node.__class__.__name__
        self._check_valid_component(
            update=update, include=include, exclude=exclude,
            module_name=component_names[0]
        )

        components = node.get_components()
        cmp0 = components[component_names[0]]
        cmp0_space = cmp0.get_hyperparameter_search_space(dataset_properties=self.dataset_properties)
        cmp0_hyperparameters = cmp0_space.get_hyperparameter_names()

        if component_names[0] == '__choice__':
            self._check_component_in_choices(
                update=update, components=components,
                include=include, exclude=exclude
            )
        elif component_names[0] not in components.keys():
            msg = _err_msg('component choice', components.keys(), component_names[0])
            raise ValueError(msg)
        elif component_names[1] not in cmp0_space and \
                not any([component_names[1] in name for name in cmp0_hyperparameters]):
            # Check if update hyperparameter is in names of hyperparameters of the search space
            # e.g. 'num_units' in 'num_units_1', 'num_units_2'
            msg = _err_msg(
                f'hyperparameter for component {cmp0.__name__} of node {node_name}',
                cmp0_hyperparameters, component_names[1]
            )
            raise ValueError(msg)

    def _check_component_in_choices(
        self,
        update: HyperparameterSearchSpaceUpdates,
        include: Optional[Dict[str, Any]],
        exclude: Optional[Dict[str, Any]],
        components: Dict[str, autoPyTorchComponent]
    ) -> None:
        """
        Check if the components in the value range of
        search space update are in components of the choice module
        """
        key_exist = {key: True for key in components.keys()}
        for choice in update.value_range:
            self._check_valid_component(
                update=update, include=include, exclude=exclude,
                module_name=choice
            )
            if not key_exist.get(choice, False):
                msg = _err_msg('component choice', components.keys(), choice)
                raise ValueError(msg)

    def _check_search_space_updates(self, include: Optional[Dict[str, Any]],
                                    exclude: Optional[Dict[str, Any]]) -> None:
        assert self.search_space_updates is not None

        key_exist = {key: True for key in self.named_steps.keys()}
        for update in self.search_space_updates.updates:
            if not key_exist.get(update.node_name, False):
                msg = _err_msg('update.node_name', self.named_steps.keys(), update.node_name)
                raise ValueError(msg)

            node = self.named_steps[update.node_name]
            node_search_space = node.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties)
            node_hyperparameters = node_search_space.get_hyperparameter_names()
            key_exist = {key: True for key in node_hyperparameters}

            if not hasattr(node, 'get_components'):
                self._check_available_components(update=update, include=include, exclude=exclude)
            elif update.hyperparameter not in node_search_space and \
                    not key_exist.get(update.hyperparameter, False):

                msg = _err_msg('hyperparameter for node', node_hyperparameters, update.hyperparameter)
                raise ValueError(msg)

    def _get_pipeline_steps(self, dataset_properties: Optional[Dict[str, Any]]
                            ) -> List[Tuple[str, autoPyTorchChoice]]:
        """
        Defines what steps a pipeline should follow.
        The step itself has choices given via autoPyTorchChoices.

        Returns:
            List[Tuple[str, autoPyTorchChoices]]: list of steps sequentially exercised
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
        fit_requirements: List[FitRequirement] = list()
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
        fit_requirements = list()  # type: List[FitRequirement]
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
