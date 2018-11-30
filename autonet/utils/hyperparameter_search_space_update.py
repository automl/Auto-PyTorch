import ast
import os

class HyperparameterSearchSpaceUpdate():
    def __init__(self, node_name, hyperparameter, value_range, log=False):
        self.node_name = node_name
        self.hyperparameter = hyperparameter
        self.value_range = value_range
        self.log = log
    
    def apply(self, pipeline, pipeline_config):
        pipeline[self.node_name]._update_hyperparameter_range(name=self.hyperparameter,
                                                              new_value_range=self.value_range,
                                                              log=self.log, 
                                                              pipeline_config=pipeline_config)

class HyperparameterSearchSpaceUpdates():
    def __init__(self, updates=[]):
        self.updates = updates
    
    def apply(self, pipeline, pipeline_config):
        for update in self.updates:
            update.apply(pipeline, pipeline_config)
    
    def append(self, node_name, hyperparameter, value_range, log=False):
        self.updates.append(HyperparameterSearchSpaceUpdate(node_name=node_name,
                                                            hyperparameter=hyperparameter,
                                                            value_range=value_range,
                                                            log=log))

    def save_as_file(self, path):
        with open(path, "w") as f:
            for update in self.updates:
                print(update.node_name, update.hyperparameter, str(update.value_range) + (" log" if update.log else ""), file=f)


def parse_hyperparameter_search_space_updates(updates_file):
    if updates_file is None or os.path.basename(updates_file) == "None":
        return None
    with open(updates_file, "r") as f:
        result = []
        for line in f:
            if line.strip() == "":
                continue
            line = line.split()
            node, hyperparameter, value_range = line[0], line[1], ast.literal_eval(line[2])
            assert isinstance(value_range, list)
            log = len(line) == 4 and "log" == line[3]
            result.append(HyperparameterSearchSpaceUpdate(node, hyperparameter, value_range, log))
    return HyperparameterSearchSpaceUpdates(result)

