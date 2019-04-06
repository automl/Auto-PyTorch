import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH


def get_hyperparameter(hyper_type, name, value_range):
    log = False
    if isinstance(value_range, tuple) and len(value_range) == 2 and isinstance(value_range[1], bool) and \
        isinstance(value_range[0], (tuple, list)):
        value_range, log = value_range

    if len(value_range) == 0:
        raise ValueError(name + ': The range has to contain at least one element')
    if len(value_range) == 1:
        return CSH.Constant(name, int(value_range[0]) if isinstance(value_range[0], bool) else value_range[0])
    if len(value_range) == 2 and value_range[0] == value_range[1]:
        return CSH.Constant(name, int(value_range[0]) if isinstance(value_range[0], bool) else value_range[0])
    if hyper_type == CSH.CategoricalHyperparameter:
        return CSH.CategoricalHyperparameter(name, value_range)
    if hyper_type == CSH.UniformFloatHyperparameter:
        assert len(value_range) == 2, "Float HP range update for %s is specified by the two upper and lower values. %s given." %(name, len(value_range))
        return CSH.UniformFloatHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    if hyper_type == CSH.UniformIntegerHyperparameter:
        assert len(value_range) == 2, "Int HP range update for %s is specified by the two upper and lower values. %s given." %(name, len(value_range))
        return CSH.UniformIntegerHyperparameter(name, lower=value_range[0], upper=value_range[1], log=log)
    raise ValueError('Unknown type: %s for hp %s' % (hyper_type, name) )

def add_hyperparameter(cs, hyper_type, name, value_range):
    return cs.add_hyperparameter(get_hyperparameter(hyper_type, name, value_range))