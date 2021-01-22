import typing
from typing import Optional

import ConfigSpace as cs


class SearchSpace:

    hyperparameter_types = {
        'categorical': cs.CategoricalHyperparameter,
        'integer': cs.UniformIntegerHyperparameter,
        'float': cs.UniformFloatHyperparameter,
        'constant': cs.Constant,
    }

    @typing.no_type_check
    def __init__(
            self,
            cs_name: str = 'Default Hyperparameter Config',
            seed: int = 11,
    ):
        """Fit the selected algorithm to the training data.

        Args:
            cs_name (str): The name of the configuration space.
            seed (int): Seed value used for the configuration space.

        Returns:
        """
        self._hp_search_space = cs.ConfigurationSpace(
            name=cs_name,
            seed=seed,
        )

    @typing.no_type_check
    def add_hyperparameter(
        self,
        name: str,
        hyperparameter_type: str,
        **kwargs,
    ):
        """Add a new hyperparameter to the configuration space.

        Args:
            name (str): The name of the hyperparameter to be added.
            hyperparameter_type (str): The type of the hyperparameter to be added.

        Returns:
            hyperparameter (cs.Hyperparameter): The hyperparameter that was added
                to the hyperparameter search space.
        """
        missing_arg = SearchSpace._assert_necessary_arguments_given(
            hyperparameter_type,
            **kwargs,
        )

        if missing_arg is not None:
            raise TypeError(f'A {hyperparameter_type} must have a value for {missing_arg}')
        else:
            hyperparameter = SearchSpace.hyperparameter_types[hyperparameter_type](
                name=name,
                **kwargs,
            )
            self._hp_search_space.add_hyperparameter(
                hyperparameter
            )

        return hyperparameter

    @staticmethod
    @typing.no_type_check
    def _assert_necessary_arguments_given(
        hyperparameter_type: str,
        **kwargs,
    ) -> Optional[str]:
        """Assert that given a particular hyperparameter type, all the
        necessary arguments are given to create the hyperparameter.

        Args:
            hyperparameter_type (str): The type of the hyperparameter to be added.

        Returns:
            missing_argument (str|None): The argument that is missing
                to create the given hyperparameter.
        """
        necessary_args = {
            'categorical': {'choices', 'default_value'},
            'integer': {'lower', 'upper', 'default', 'log'},
            'float': {'lower', 'upper', 'default', 'log'},
            'constant': {'value'},
        }

        hp_necessary_args = necessary_args[hyperparameter_type]
        for hp_necessary_arg in hp_necessary_args:
            if hp_necessary_arg not in kwargs:
                return hp_necessary_arg

        return None

    @typing.no_type_check
    def set_parent_hyperperparameter(
            self,
            child_hp,
            parent_hp,
            parent_value,
    ):
        """Activate the child hyperparameter on the search space only if the
        parent hyperparameter takes a particular value.

        Args:
            child_hp (cs.Hyperparameter): The child hyperparameter to be added.
            parent_hp (cs.Hyperparameter): The parent hyperparameter to be considered.
            parent_value (str|float|int): The value of the parent hyperparameter for when the
                child hyperparameter will be added to the search space.

        Returns:
        """
        self._hp_search_space.add_condition(
            cs.EqualsCondition(
                child_hp,
                parent_hp,
                parent_value,
            )
        )

    @typing.no_type_check
    def add_configspace_condition(
        self,
        child_hp,
        parent_hp,
        configspace_condition,
        value,
    ):
        """Add a condition on the chi

        Args:
            child_hp (cs.Hyperparameter): The child hyperparameter to be added.
            parent_hp (cs.Hyperparameter): The parent hyperparameter to be considered.
            configspace_condition (cs.AbstractCondition): The condition to be fullfilled
                by the parent hyperparameter. A list of all the possible conditions can be
                found at ConfigSpace/conditions.py.
            value (str|float|int|list): The value of the parent hyperparameter to be matched
                in the condition. value needs to be a list only for the InCondition.

        Returns:
        """
        self._hp_search_space.add_condition(
            configspace_condition(
                child_hp,
                parent_hp,
                value,
            )
        )
