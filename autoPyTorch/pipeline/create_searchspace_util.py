import itertools
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.forbidden import ForbiddenAndConjunction
from ConfigSpace.forbidden import ForbiddenEqualsClause

import numpy as np

from autoPyTorch.pipeline.components.base_choice import autoPyTorchChoice
from autoPyTorch.pipeline.components.base_component import autoPyTorchComponent


def get_match_array(
    pipeline: List[Tuple[str, autoPyTorchChoice]],
    dataset_properties: Dict[str, Any],
    include: Optional[Dict[str, Any]] = None,
    exclude: Optional[Dict[str, Any]] = None
) -> np.ndarray:

    # Duck typing, not sure if it's good...
    node_i_is_choice = []
    node_i_choices: List[List[Union[autoPyTorchComponent, autoPyTorchChoice]]] = []
    node_i_choices_names = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices_names.append(list(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).keys()))
            node_i_choices.append(list(node.get_available_components(
                dataset_properties, include=node_include,
                exclude=node_exclude).values()))

        else:
            node_i_choices.append([node])

    matches_dimensions = [len(choices) for choices in node_i_choices]
    # Start by allowing every combination of nodes. Go through all
    # combinations/pipelines and erase the illegal ones
    matches = np.ones(matches_dimensions, dtype=np.int32)

    # TODO: Check if we need this, like are there combinations from the
    # pipeline we should dynamically avoid?
    return matches


def find_active_choices(
    matches: np.ndarray,
    node: Union[autoPyTorchComponent, autoPyTorchChoice],
    node_idx: int,
    dataset_properties: Dict[str, Any],
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> List[str]:
    if not hasattr(node, "get_available_components"):
        raise ValueError()
    available_components = node.get_available_components(dataset_properties,
                                                         include=include,
                                                         exclude=exclude)
    assert matches.shape[node_idx] == len(available_components), \
        (matches.shape[node_idx], len(available_components))

    choices = []
    for c_idx, component in enumerate(available_components):
        slices = tuple(slice(None) if idx != node_idx else slice(c_idx, c_idx + 1)
                       for idx in range(len(matches.shape)))

        if np.sum(matches[slices]) > 0:
            choices.append(component)
    return choices


def add_forbidden(
    conf_space: ConfigurationSpace,
    pipeline: List[Tuple[str, autoPyTorchChoice]],
    matches: np.ndarray,
    dataset_properties: Dict[str, Any],
    include: Optional[Dict[str, Any]] = None,
    exclude: Optional[Dict[str, Any]] = None
) -> ConfigurationSpace:
    # Not sure if this works for 3D
    node_i_is_choice = []
    node_i_choices_names: List[List[str]] = []
    node_i_choices: List[List[Union[autoPyTorchComponent, autoPyTorchChoice]]] = []
    all_nodes = []
    for node_name, node in pipeline:
        all_nodes.append(node)
        is_choice = hasattr(node, "get_available_components")
        node_i_is_choice.append(is_choice)

        node_include = include.get(
            node_name) if include is not None else None
        node_exclude = exclude.get(
            node_name) if exclude is not None else None

        if is_choice:
            node_i_choices_names.append(
                [str(element) for element in
                    node.get_available_components(
                    dataset_properties, include=node_include,
                    exclude=node_exclude).keys()]

            )
            node_i_choices.append(
                list(node.get_available_components(
                    dataset_properties, include=node_include,
                    exclude=node_exclude
                ).values()))

        else:
            node_i_choices_names.append([node_name])
            node_i_choices.append([node])

    # Find out all chains of choices. Only in such a chain its possible to
    # have several forbidden constraints
    choices_chains = []
    idx = 0
    while idx < len(pipeline):
        if node_i_is_choice[idx]:
            chain_start = idx
            idx += 1
            while idx < len(pipeline) and node_i_is_choice[idx]:
                idx += 1
            chain_stop = idx
            choices_chains.append((chain_start, chain_stop))
        idx += 1

    for choices_chain in choices_chains:
        constraints: Set[Tuple] = set()

        chain_start = choices_chain[0]
        chain_stop = choices_chain[1]
        chain_length = chain_stop - chain_start

        # Add one to have also have chain_length in the range
        for sub_chain_length in range(2, chain_length + 1):
            for start_idx in range(chain_start, chain_stop - sub_chain_length + 1):
                indices = range(start_idx, start_idx + sub_chain_length)
                node_names = [pipeline[idx][0] for idx in indices]

                num_node_choices = []
                node_choice_names = []
                skip_array_shape = []

                for idx in indices:
                    node = all_nodes[idx]
                    available_components = node.get_available_components(
                        dataset_properties,
                        include=node_i_choices_names[idx])
                    assert len(available_components) > 0, len(available_components)
                    skip_array_shape.append(len(available_components))
                    num_node_choices.append(range(len(available_components)))
                    node_choice_names.append([name for name in available_components])

                # Figure out which choices were already abandoned
                skip_array = np.zeros(skip_array_shape)
                for product in itertools.product(*num_node_choices):
                    for node_idx, choice_idx in enumerate(product):
                        node_idx += start_idx
                        slices_ = tuple(
                            slice(None) if idx != node_idx else
                            slice(choice_idx, choice_idx + 1) for idx in
                            range(len(matches.shape)))

                        if np.sum(matches[slices_]) == 0:
                            skip_array[product] = 1

                for product in itertools.product(*num_node_choices):
                    if skip_array[product]:
                        continue

                    slices = tuple(
                        slice(None) if idx not in indices else
                        slice(product[idx - start_idx],
                              product[idx - start_idx] + 1) for idx in
                        range(len(matches.shape)))

                    if np.sum(matches[slices]) == 0:
                        constraint = tuple([(node_names[i],
                                             node_choice_names[i][product[i]])
                                            for i in range(len(product))])

                        # Check if a more general constraint/forbidden clause
                        #  was already added
                        continue_ = False
                        for constraint_length in range(2, len(constraint)):
                            constr_starts = len(constraint) - constraint_length + 1
                            for constraint_start_idx in range(constr_starts):
                                constraint_end_idx = constraint_start_idx + constraint_length
                                sub_constraint = constraint[constraint_start_idx:constraint_end_idx]
                                if sub_constraint in constraints:
                                    continue_ = True
                                    break
                            if continue_:
                                break
                        if continue_:
                            continue

                        constraints.add(constraint)

                        forbiddens = []
                        for i in range(len(product)):
                            forbiddens.append(
                                ForbiddenEqualsClause(conf_space.get_hyperparameter(
                                    node_names[i] + ":__choice__"),
                                    node_choice_names[i][product[i]]))
                        forbidden = ForbiddenAndConjunction(*forbiddens)
                        conf_space.add_forbidden_clause(forbidden)

    return conf_space
