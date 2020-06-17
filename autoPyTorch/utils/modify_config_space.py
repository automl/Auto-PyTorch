import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
import copy

def remove_constant_hyperparameter(cs):
    constants = dict()

    hyperparameter_to_add = []
    for hyper in cs.get_hyperparameters():
        const, value = is_constant(hyper)
        if const:
            constants[hyper.name] = value
        else:
            hyperparameter_to_add.append(copy.copy(hyper))
    

    for name in constants:
        hp = cs.get_hyperparameter(name)
        truncate_hyperparameter(cs, hp)

    cs._hyperparameter_idx = dict()
    cs._idx_to_hyperparameter = dict()
    cs._sort_hyperparameters()
    cs._update_cache()
        
    return cs, constants


def is_constant(hyper):
    if isinstance(hyper, CSH.Constant):
        return True, hyper.value

    elif isinstance(hyper, CSH.UniformFloatHyperparameter) or isinstance(hyper, CSH.UniformIntegerHyperparameter):
        if abs(hyper.upper - hyper.lower) < 1e-10:
            return True, hyper.lower
        
    elif isinstance(hyper, CSH.CategoricalHyperparameter):
        if len(hyper.choices) == 1:
            return True, hyper.choices[0]
        
    return False, None


def override_hyperparameter(config_space, hyper):
    import ConfigSpace.conditions as CSC

    for condition in config_space._children[hyper.name].values():
        subconditions = condition.components if isinstance(condition, CSC.AbstractConjunction) else [condition]
        for subcondition in subconditions:
            if subcondition.parent.name == hyper.name:
                subcondition.parent = hyper

    for condition in config_space._parents[hyper.name].values():
        if condition is None:
            continue # root
        subconditions = condition.components if isinstance(condition, CSC.AbstractConjunction) else [condition]
        for subcondition in subconditions:
            if subcondition.child.name == hyper.name:
                subcondition.child = hyper

    config_space._hyperparameters[hyper.name] = hyper


def update_conditions(config_space, parent):
    import ConfigSpace.conditions as CSC

    if parent.name not in config_space._hyperparameters:
        # already removed -> all condition already updated
        return

    possible_values, is_value_range = get_hyperparameter_values(parent)
    children = [config_space.get_hyperparameter(name) for name in config_space._children[parent.name]]

    for child in children:
        if child.name not in config_space._children[parent.name]:
            # already cut
            continue
        condition = config_space._children[parent.name][child.name]

        if isinstance(condition, CSC.AbstractConjunction):
            is_and = isinstance(condition, CSC.AndConjunction)
            state = 2
            
            new_subconditions = []
            for subcondition in condition.components:
                if subcondition.parent.name != parent.name:
                    new_subconditions.append(subcondition)
                    continue
                substate = get_condition_state(subcondition, possible_values, is_value_range)
                if substate == 0 and is_and and state == 2:
                    state = 0

                if substate == 1 and not is_and and state == 2:
                    state = 1

                if substate == 2:
                    new_subconditions.append(subcondition)
                
                else:
                    # condition is not relevant anymore
                    del config_space._children[parent.name][child.name]
                    del config_space._parents[child.name][parent.name]
                    for grand_parent, cond in config_space._parents[parent.name].items():
                        if cond is None:
                            continue
                        cond_type = type(cond)
                        values, _ = get_hyperparameter_values(cond.parent)
                        # fake parent value first as it might be invalid atm and gets truncated later
                        new_condition = cond_type(child, cond.parent, values[0])
                        new_condition.value = cond.value
                        config_space._children[grand_parent][child.name] = new_condition
                        config_space._parents[child.name][grand_parent] = new_condition

            if len(new_subconditions) == 0:
                state = 1 if is_and else 0 # either everything was false or true

            if state == 2:

                if len(new_subconditions) == 1:
                    condition = new_subconditions[0]
                    config_space._children[condition.parent.name][child.name] = new_subconditions[0]
                    config_space._parents[child.name][condition.parent.name] = new_subconditions[0]
                else:
                    condition.__init__(*new_subconditions)

                    for subcondition in new_subconditions:
                        config_space._children[subcondition.parent.name][child.name] = condition
                        config_space._parents[child.name][subcondition.parent.name] = condition

        else:
            state = get_condition_state(condition, possible_values, is_value_range)

        if state == 1:
            del config_space._children[parent.name][child.name]
            del config_space._parents[child.name][parent.name]

            for grand_parent, cond in config_space._parents[parent.name].items():
                if cond is None:
                    continue
                cond_type = type(cond)
                values, _ = get_hyperparameter_values(cond.parent)
                # fake parent value first as it might be invalid atm and gets truncated later
                new_condition = cond_type(child, cond.parent, values[0])
                new_condition.value = cond.value
                config_space._children[grand_parent][child.name] = new_condition
                config_space._parents[child.name][grand_parent] = new_condition

            if len(config_space._parents[child.name]) == 0:
                config_space._conditionals.remove(child.name)
        if state == 0:
            truncate_hyperparameter(config_space, child)



    
def truncate_hyperparameter(config_space, hyper):
    if hyper.name not in config_space._hyperparameters:
        return

    parent_names = list(config_space._parents[hyper.name].keys())
    for parent_name in parent_names:
        del config_space._children[parent_name][hyper.name]

    del config_space._parents[hyper.name]
    del config_space._hyperparameters[hyper.name]

    if hyper.name in config_space._conditionals:
        config_space._conditionals.remove(hyper.name)

    child_names = list(config_space._children[hyper.name].keys())
    for child_name in child_names:
        truncate_hyperparameter(config_space, config_space.get_hyperparameter(child_name))


def get_condition_state(condition, possible_values, is_range):
    """
        0: always false
        1: always true
        2: true or false
    """
    import ConfigSpace.conditions as CSC

    c_val = condition.value
    if isinstance(condition, CSC.EqualsCondition):
        if is_range:
            if approx(possible_values[0], possible_values[1]):
                return 1 if approx(possible_values[0], c_val) else 0
            return 2 if c_val >= possible_values[0] and c_val <= possible_values[1] else 0
        else:
            if len(possible_values) == 1:
                return 1 if c_val == possible_values[0] else 0
            return 2 if c_val in possible_values else 0
            
    if isinstance(condition, CSC.NotEqualsCondition):
        if is_range:
            if approx(possible_values[0], possible_values[1]):
                return 0 if approx(possible_values[0], c_val) else 1
            return 2 if c_val >= possible_values[0] and c_val <= possible_values[1] else 1
        else:
            if len(possible_values) == 1:
                return 0 if c_val == possible_values[0] else 1
            return 2 if c_val in possible_values else 1

    if isinstance(condition, CSC.GreaterThanCondition): # is_range has to be true
        if c_val < possible_values[0]:
            return 1
        if c_val >= possible_values[1]:
            return 0
        return 2

    if isinstance(condition, CSC.LessThanCondition): # is_range has to be true
        if c_val <= possible_values[0]:
            return 0
        if c_val > possible_values[1]:
            return 1
        return 2

    if isinstance(condition, CSC.InCondition):
        inter = set(possible_values).intersection(set(c_val))
        if len(inter) == len(possible_values):
            return 1
        if len(inter) == 0:
            return 0
        return 2
        

def approx(x, y):
    return abs(x - y) < 1e-10

def get_hyperparameter_values(hyper):
    """Returns list[choices/range] and bool[is value range]
    """
    import ConfigSpace.hyperparameters as CSH

    if isinstance(hyper, CSH.CategoricalHyperparameter):
        return hyper.choices, False

    if isinstance(hyper, CSH.NumericalHyperparameter):
        return [hyper.lower, hyper.upper], True
        
    if isinstance(hyper, CSH.Constant):
        return [hyper.value, hyper.value], True

    raise ValueError(str(type(hyper)) + ' is not supported')
