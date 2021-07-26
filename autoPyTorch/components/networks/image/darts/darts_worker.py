import os
import time
import argparse
#from copy import copy, deepcopy

import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH
from hpbandster.core.worker import Worker
# from .helper import darts_cifar10


PRIMITIVES = [
    #'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
]


class DARTSWorker(Worker):
    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     #self.darts_mainsourcepath = '/home/zelaa/Thesis/bohb-darts/workers/lib'
    #     self.darts_path = os.getcwd() + '/workers/lib/darts_space'

    # def compute(self, config, budget, config_id, working_directory):
    #     return darts_cifar10(config=config,
    #                          budget=int(budget),
    #                          config_id=config_id,
    #                          directory=working_directory,
    #                          darts_source=self.darts_path)

    @staticmethod
    def get_config_space():
        config_space = CS.ConfigurationSpace()

        # here we instantiate one categorical hyperparameter for each edge in
        # the DARTS cell
        for i in range(14):
            config_space.add_hyperparameter(CSH.CategoricalHyperparameter('edge_normal_{}'.format(i),
                                                                          PRIMITIVES))
            config_space.add_hyperparameter(CSH.CategoricalHyperparameter('edge_reduce_{}'.format(i),
                                                                          PRIMITIVES))
        # for the intermediate node 2 we add directly the two incoming edges to
        # the config_space. All nodes are topologicaly sorted and the labels 0
        # and 1 correspond to the 2 input nodes of the cell. nodes 2, 3, 4, 5
        # are intermediate nodes. We define below a CategoricalHyperparameter
        # for nodes 3, 4, 5 with each category representing two possible
        # predecesor nodes indices (for node 2 there is only one possibility)
        pred_nodes = {'3': ['0_1', '0_2', '1_2'],
                      '4': ['0_1', '0_2', '0_3', '1_2', '1_3', '2_3'],
                      '5': ['0_1', '0_2', '0_3', '0_4', '1_2', '1_3', '1_4',
                            '2_3', '2_4', '3_4']
                     }

        for i in range(3, 6):
            config_space.add_hyperparameter(CSH.CategoricalHyperparameter('inputs_node_normal_{}'.format(i),
                                                                          pred_nodes[str(i)]))
            config_space.add_hyperparameter(CSH.CategoricalHyperparameter('inputs_node_reduce_{}'.format(i),
                                                                          pred_nodes[str(i)]))

        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('layers', lower=5, upper=20))
        config_space.add_hyperparameter(CSH.UniformIntegerHyperparameter('init_channels', lower=8, upper=50, log=True))
        config_space.add_hyperparameter(CSH.Constant('drop_path_prob', 0.2))
        config_space.add_hyperparameter(CSH.CategoricalHyperparameter('auxiliary', [True, False]))

        # now we define the conditions constraining the inclusion of the edges
        # on the optimization in order to be consistent with the DARTS original
        # search space
        for cell_type in ['normal', 'reduce']:
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_2'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_3'.format(cell_type)),
                                                      values=['0_1', '0_2']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_3'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_3'.format(cell_type)),
                                                      values=['0_1', '1_2']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_4'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_3'.format(cell_type)),
                                                      values=['0_2', '1_2']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_5'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_4'.format(cell_type)),
                                                      values=['0_1', '0_2', '0_3']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_6'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_4'.format(cell_type)),
                                                      values=['0_1', '1_2', '1_3']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_7'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_4'.format(cell_type)),
                                                      values=['0_2', '1_2', '2_3']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_8'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_4'.format(cell_type)),
                                                      values=['0_3', '1_3', '2_3']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_9'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_5'.format(cell_type)),
                                                      values=['0_1', '0_2', '0_3', '0_4']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_10'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_5'.format(cell_type)),
                                                      values=['0_1', '1_2', '1_3', '1_4']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_11'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_5'.format(cell_type)),
                                                      values=['0_2', '1_2', '2_3', '2_4']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_12'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_5'.format(cell_type)),
                                                      values=['0_3', '1_3', '2_3', '3_4']))
            config_space.add_condition(CS.InCondition(child=config_space.get_hyperparameter('edge_{}_13'.format(cell_type)),
                                                      parent=config_space.get_hyperparameter('inputs_node_{}_5'.format(cell_type)),
                                                      values=['0_4', '1_4', '2_4', '3_4']))

        return config_space
