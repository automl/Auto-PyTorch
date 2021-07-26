import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_hyperparameter(hyper_type, name, value_range, log=False):
    if len(value_range) == 0:
        raise ValueError(name + ': The range has to contain at least one element')
    if len(value_range) == 1:
        return CSH.Constant(name, value_range[0])
    if len(value_range) == 2 and value_range[0] == value_range[1]:
        return CSH.Constant(name, value_range[0])
    if hyper_type == CSH.CategoricalHyperparameter:
        return CSH.CategoricalHyperparameter(name, value_range)
    if hyper_type == CSH.UniformFloatHyperparameter:
        return CSH.UniformFloatHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    if hyper_type == CSH.UniformIntegerHyperparameter:
        return CSH.UniformIntegerHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    raise ValueError('Unknown type: ' + str(hyper_type))

def add_hyperparameter(cs, hyper_type, name, value_range, log=False):
    return cs.add_hyperparameter(get_hyperparameter(hyper_type, name, value_range, log))