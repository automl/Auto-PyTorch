import ast
import os
from typing import List

class HyperparameterSearchSpaceUpdate():
    def __init__(self, node_name, hyperparameter, value_range, default_value, log=False):
        self.node_name = node_name
        self.hyperparameter = hyperparameter
        self.value_range = value_range
        self.log = log
        self.default_value = default_value

    def apply(self, pipeline):
        [node[1]._apply_search_space_update(name=self.hyperparameter, new_value_range=self.value_range, log=self.log,
                                            default_value=self.default_value) for node in pipeline if node[0] == self.node_name]


class HyperparameterSearchSpaceUpdates():
    def __init__(self, updates: List[HyperparameterSearchSpaceUpdate] = []):
        self.updates = updates

    def apply(self, pipeline):
        for update in self.updates:
            update.apply(pipeline)

    def append(self, node_name, hyperparameter, value_range, default_value, log=False):
        self.updates.append(HyperparameterSearchSpaceUpdate(node_name=node_name,
                                                            hyperparameter=hyperparameter,
                                                            value_range=value_range,
                                                            default_value=default_value,
                                                            log=log))

    def save_as_file(self, path):
        with open(path, "w") as f:
            for update in self.updates:
                print(update.node_name, update.hyperparameter, str(update.value_range),
                      update.default_value + (" log" if update.log else ""),
                      file=f)


def parse_hyperparameter_search_space_updates(updates_file):
    if updates_file is None or os.path.basename(updates_file) == "None":
        return None
    with open(updates_file, "r") as f:
        result = []
        for line in f:
            if line.strip() == "":
                continue
            line = line.split()
            node, hyperparameter, value_range, default_value = line[0], line[1], ast.literal_eval(line[2]), line[3]
            assert isinstance(value_range, list)
            log = len(line) == 5 and "log" == line[4]
            result.append(HyperparameterSearchSpaceUpdate(node, hyperparameter, value_range, default_value, log))
    return HyperparameterSearchSpaceUpdates(result)
