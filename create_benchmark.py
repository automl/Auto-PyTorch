import argparse
import os as os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from IPython import embed

from tensorflow.python.summary.summary_iterator import summary_iterator
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


TASK_DICT = {
    "189356" : "albert",
    "189355" : "dionis",
    "189354" : "airlines",
    "168912" : "sylvine",
    "168911" : "jasmine",
    "168910" : "fabert",
    "168909" : "dilbert",
    "168908" : "christine",
    "168868" : "APSFailure",
    "168338" : "riccardo",
    "168337" : "guillermo",
    "168335" : "MiniBooNE",
    "168332" : "robert",
    "168331" : "volkert",
    "168330" : "jannis",
    "168329" : "helena",
    "167120" : "numerai28.6",
    "167119" : "jungle_chess_2pcs_raw_endgame_complete",
    "146825" : "Fashion-MNIST",
    "146822" : "segment",
    "146821" : "car",
    "146818" : "Australian",
    "146606" : "higgs",
    "146212" : "shuttle",
    "146195" : "connect-4",
    "34539" : "Amazon_employee_access",
    "14965" : "bank-marketing",
    "10101" : "blood-transfusion-service-center",
    "9981" : "cnae-9",
    "9977" : "nomao",
    "9952" : "phoneme",
    "7593" : "covertype",
    "7592" : "adult",
    "3945" : "KDDCup09_appetency",
    "53" : "vehicle",
    "3917" : "kc1",
    "12" : "mfeat-factors",
    "31" : "credit-g",
    "3" : "kr-vs-kp",
    "167149" : "kr-vs-kp",
    "167152" : "mfeat-factors",
    "167161" : "credit-g",
    "167168" : "vehicle",
    "167181" : "kc1",
    "126025" : "adult",
    "167190" : "phoneme",
    "126026" : "nomao",
    "167185" : "cnae-9",
    "167184" : "blood-transfusion-service-center",
    "126029" : "bank-marketing",
    "167201" : "connect-4",
    "189905" : "car",
    "189906" : "segment",
    "189908" : "Fashion-MNIST",
    "189909" : "jungle_chess_2pcs_raw_endgame_complete",
    "167083" : "numerai28.6",
    "189873" : "dionis",
    "167104" : "Australian",
    "189862" : "jasmine",
    "189865" : "sylvine",
    "189866" : "albert",
    "167200" : "higgs"
}


def get_subdirs(root, prefix=None):
    if prefix is not None:
        return [os.path.join(root, p) for p in os.listdir(root) if p.startswith(prefix)]
    else:
        return [os.path.join(root, p) for p in os.listdir(root)]


class ResultsReader(object):
    
    def __init__(self, json_dir, tb_log_dir, info_dir):
        self.json_dir = json_dir
        self.data = self.read_json(json_dir)
        self.info = self.read_json(info_dir)
        
        self.tb_log_dir = tb_log_dir
        self.accumulator = EventAccumulator(tb_log_dir)
        self.accumulator.Reload()
        
        self.tags = self.get_scalar_tags()
        self.start_time = self.get_start_time()

    def read_json(self, json_dir):
        with open(json_dir, "r") as f:
            data = json.load(f)
        return data
    
    def get_hyperpar_config(self):
        hyperpar_config = self.data["optimized_hyperparameter_config"]
        drop_keys = ['InitializationSelector:initialization_method',
                    'InitializationSelector:initializer:initialize_bias',
                    'TrainNode:batch_loss_computation_technique',
                    'ResamplingStrategySelector:over_sampling_method',
                    'ResamplingStrategySelector:target_size_strategy',
                    'ResamplingStrategySelector:under_sampling_method',
                    'PreprocessorSelector:preprocessor',
                    'NetworkSelector:shapedmlpnet:use_dropout']
        change_keys = {'CreateDataLoader:batch_size':'batch_size',
                       'Imputation:strategy':'imputation_strategy',
                       'LearningrateSchedulerSelector:lr_scheduler':'learning_rate_scheduler',
                       'LossModuleSelector:loss_module':'loss',
                       'NetworkSelector:network':'network',
                       'NetworkSelector:shapedmlpnet:max_dropout':'max_dropout',
                       'NormalizationStrategySelector:normalization_strategy':'normalization_strategy',
                       'OptimizerSelector:optimizer':'optimizer',
                       'LearningrateSchedulerSelector:cosine_annealing:T_max':"cosine_annealing_T_max",
                       'LearningrateSchedulerSelector:cosine_annealing:eta_min':"cosine_annealing_eta_min",
                       'NetworkSelector:shapedmlpnet:activation':'activation',
                       'NetworkSelector:shapedmlpnet:max_units':'max_units',
                       'NetworkSelector:shapedmlpnet:mlp_shape':'mlp_shape',
                       'NetworkSelector:shapedmlpnet:num_layers':'num_layers',
                       'OptimizerSelector:sgd:learning_rate':'learning_rate',
                       'OptimizerSelector:sgd:momentum':'momentum',
                       'OptimizerSelector:sgd:weight_decay':'weight_decay'}
        for key in drop_keys:
            del hyperpar_config[key]
        for key in change_keys:
            val = hyperpar_config[key]
            new_key = change_keys[key]
            del hyperpar_config[key]
            hyperpar_config[new_key] = val
        return self.data["optimized_hyperparameter_config"]
    
    def get_results(self):
        results = dict()
        results["model_parameters"] = int(self.data["info"]["model_parameters"])
        results["final_train_cross_entropy"] = self.data["info"]["loss"]
        results["final_train_accuracy"] = self.data["info"]["train_accuracy"]
        results["final_train_balanced_accuracy"] = self.data["info"]["train_balanced_accuracy"]
        results["final_val_cross_entropy"] = self.data["info"]["val_cross_entropy"]
        results["final_val_accuracy"] = self.data["info"]["val_accuracy"]
        results["final_val_balanced_accuracy"] = self.data["info"]["val_balanced_accuracy"]
        results["final_test_cross_entropy"] = self.data["info"]["test_cross_entropy"]
        results["final_test_accuracy"] = self.data["info"]["test_result"]
        results["final_test_balanced_accuracy"] = self.data["info"]["test_balanced_accuracy"]
        #results["gradient_norm"] = self.data["info"]["gradient_norm"]
        #results["gradient_mean"] = self.data["info"]["gradient_mean"]
        #results["gradient_std"] = self.data["info"]["gradient_std"]
        return results
    
    def get_full_json(self):
        return self.data
    
    def get_info(self):
        return self.info
    
    def get_scalar_tags(self):
        return self.accumulator.Tags()["scalars"]
    
    def get_start_time(self):
        return self.accumulator.FirstEventTimestamp()
    
    def read_scalar_data(self, exclude_gradient_data=True):
        uninteresting = ['Train/step', 'Train/budget', 'Train/lr_scheduler_converged', 'Train/epoch', 'Train/model_parameters']
        
        if exclude_gradient_data:
            gradient_tags = [t for t in self.tags if "gradient" in t]
            uninteresting = uninteresting + gradient_tags
        
        scalar_event_dict = {key:[] for key in self.tags if key not in uninteresting}

        
        epochs = []
        for event in self.accumulator.Scalars('Train/epoch'):
            epochs.append((event.wall_time-self.start_time, event.value))
        
        for tag in self.tags:
            if tag not in uninteresting:
                if tag=="Train/lr":
                    scalar_event_dict[tag].append((epochs[0][0], epochs[0][1], 0))
                    for ind, event in enumerate(self.accumulator.Scalars(tag)):
                        scalar_event_dict[tag].append((epochs[ind+1][0], epochs[ind+1][1], event.value))
                else:
                    for ind, event in enumerate(self.accumulator.Scalars(tag)):
                        scalar_event_dict[tag].append((epochs[ind][0], epochs[ind][1], event.value))
                    
        return scalar_event_dict

def convert_tbdata(tb_data):
    converted = dict()
    converted["time"] = [t for t,e,v in tb_data["Train/loss"]]
    converted["epoch"] = [int(e) for t,e,v in tb_data["Train/loss"]]
    for key, val_list in tb_data.items():
        converted[key] = [v for t,e,v in val_list]
    return converted


def add_results(full_dict, config_hashdict, config, tb_logs, results, info, seed, budget):

    config_without_tmax = {k:v for k,v in config.items() if not "T_max" in k}
    config_hash = hash(config_without_tmax.__repr__())
    
    if config_hash in config_hashdict.keys():
        config_id = config_hashdict[config_hash]
    else:
        config_id = len(config_hashdict)
        config_hashdict[config_hash] = config_id

    dataset_name = TASK_DICT[str(info["OpenML_task_id"])]

    if not config_id in full_dict[dataset_name].keys():
        full_dict[dataset_name][config_id] = dict()

    if not budget in full_dict[dataset_name][config_id].keys():
        full_dict[dataset_name][config_id][budget] = {"config" : config,
                                                      "results" : {},
                                                      "log": {}}

    full_dict[dataset_name][config_id][budget]["results"][seed] = {**results, **info}
    full_dict[dataset_name][config_id][budget]["log"][seed] = convert_tbdata(tb_logs)


if __name__=="__main__":
    additional_jsons_dir = []
    tb_log_root_dir = "/home/zimmerl/APT_step_logging/Auto-PyTorch/logs/shapedmlp_2k/"
    save_dir = "bench.json"

    budget_dirs = get_subdirs(tb_log_root_dir, prefix="budget")
    
    config_hashdict = dict()
    full_dict = {dataset_name:{} for dataset_name in np.unique(list(TASK_DICT.values()))}

    # Loop through budget dirs
    for bd in budget_dirs:
        budget = bd.split("_")[-1]
        seed_dirs = get_subdirs(bd, prefix="seed")

        # Loop through seed dirs
        for sd in seed_dirs:
            seed = sd.split("_")[-1]
            print("Budget %s, seed %s" %(budget, seed))
            rundirs = get_subdirs(sd, prefix="run")

            # Loop through rundirs
            for rundir in tqdm(rundirs):
                
                # Get results and add to json
                try:
                    tb_logdir, info_dir, results_dir = sorted(get_subdirs(rundir))
                except Exception as e:
                    if isinstance(e, ValueError):
                        continue
                    else:
                        raise e
                reader = ResultsReader(results_dir, tb_logdir, info_dir)

                config = reader.get_hyperpar_config()
                tb_logs = reader.read_scalar_data()
                results = reader.get_results()
                info = reader.get_info()

                add_results(full_dict, config_hashdict, config, tb_logs, results, info, seed, budget)

            # NOTE: This saves every step
            #logdir = "bench_"+str(budget)+"_"+str(seed)+".json"
            #with open(save_dir, "w") as f:
            #    json.dump(full_dict, f)
            #full_dict = {dataset_name:{} for dataset_name in np.unique(list(TASK_DICT.values()))}

            with open(save_dir, "w") as f:
                json.dump(full_dict, f)


    # Save
    with open(save_dir, "w") as f:
        json.dump(full_dict, f)
