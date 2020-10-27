import os
import json
import time
import argparse
import numpy as np

from sklearn.preprocessing import OneHotEncoder as OHE

from autoPyTorch.components.metrics import accuracy, cross_entropy, auc_metric
from autoPyTorch.pipeline.nodes.metric_selector import AutoNetMetric, undo_ohe, default_minimize_transform
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection
from hpbandster.core.result import logged_results_to_HBS_result


def transform_to_probabilities(labels):
    if labels == []:
        return []
    classes = sorted(np.unique(labels))

    encoded = np.zeros((len(labels),len(classes)))

    for ind, lab in enumerate(labels):
        encoded[ind][lab] = 1

    return encoded
    

class EnsembleTrajectorySimulator():

    def __init__(self, ensemble_pred_dir, ensemble_config, seed):
        
        self.ensemble_pred_dir = os.path.join(ensemble_pred_dir, "predictions_for_ensemble.npy")
        self.ensemble_pred_dir_test = os.path.join(ensemble_pred_dir, "test_predictions_for_ensemble.npy")
        self.ensemble_config = ensemble_config
        self.seed = seed

        self.read_runfiles()
        self.timesteps = self.get_timesteps()
        
        self.ensemble_selection = EnsembleSelection(**ensemble_config)

    def read_runfiles(self, val_split=0.5):

        self.ensemble_identifiers = []
        self.ensemble_predictions = []
        self.ensemble_predictions_ensemble_val = []
        self.ensemble_timestamps = []

        with open(self.ensemble_pred_dir, "rb") as f:
            self.labels = np.load(f, allow_pickle=True)

            if val_split is not None and val_split>0:
                # shuffle val data
                indices = np.arange(len(self.labels))
                rng = np.random.default_rng(seed=self.seed)
                rng.shuffle(indices)

                # Create a train val split for the ensemble from the validation data
                split = int(len(indices) * (1-val_split))
                self.train_indices = indices[:split]
                self.val_indices = indices[split:]
                
                self.labels_ensemble_val = self.labels[self.val_indices]
                self.labels = self.labels[self.train_indices]
            else:
                self.labels_ensemble_val = []

            while True:
                try:
                    job_id, budget, timestamps = np.load(f, allow_pickle=True)
                    predictions = np.array(np.load(f, allow_pickle=True))
                    
                    if val_split is not None and val_split>0:
                        self.ensemble_identifiers.append(job_id + (budget, ))
                        self.ensemble_predictions.append(predictions[self.train_indices])
                        self.ensemble_predictions_ensemble_val.append(predictions[self.val_indices])
                        self.ensemble_timestamps.append(timestamps)
                    else:
                        self.ensemble_identifiers.append(job_id + (budget, ))
                        self.ensemble_predictions.append(predictions)
                        self.ensemble_timestamps.append(timestamps)
                except (EOFError, OSError):
                    break

        self.ensemble_predictions_test = []
        self.test_labels = None

        if os.path.exists(self.ensemble_pred_dir_test):
            with open(self.ensemble_pred_dir_test, "rb") as f:
                try:
                    self.test_labels = np.load(f, allow_pickle=True)
                except (EOFError, OSError):
                    pass

                while True:
                    try:
                        job_id, budget, timestamps = np.load(f, allow_pickle=True)
                        predictions = np.array(np.load(f, allow_pickle=True))

                        self.ensemble_predictions_test.append(predictions)
                        #print("==> Adding test labels with shape", predictions.shape)
                    except (EOFError, OSError):
                        break

        # Transform timestamps to start at t=0
        self.transform_timestamps(add_time=-self.ensemble_timestamps[0]["submitted"])

        print("==> Found %i val preds" %len(self.ensemble_predictions))
        print("==> Found %i test preds" %len(self.ensemble_predictions_test))
        print("==> Found %i timestamps" %len(self.ensemble_timestamps))

        
        if len(np.array(self.labels).shape) < len(self.ensemble_predictions[0].shape):
            self.labels = transform_to_probabilities(self.labels)
            self.labels_ensemble_val = transform_to_probabilities(self.labels_ensemble_val)
            if self.test_labels is not None:
                self.test_labels = transform_to_probabilities(self.test_labels)

    def transform_timestamps(self, add_time):
        transformed_timestamps = [t["finished"]+add_time for t in self.ensemble_timestamps]
        self.ensemble_timestamps = transformed_timestamps

    def get_timesteps(self):
        # we want at least 2 models
        first_timestep = self.ensemble_timestamps[1]
        final_timestep = self.ensemble_timestamps[-1]
        return self.ensemble_timestamps[1:]

    def get_ensemble_performance(self, timestep):
        cutoff_ind = np.argmin([abs(t - timestep) for t in self.ensemble_timestamps])+1
        print("==> Considering %i models and timestep %f" %(cutoff_ind, timestep))

        # create ensemble
        self.ensemble_selection.fit(np.array(self.ensemble_predictions[0:cutoff_ind]), self.labels, self.ensemble_identifiers[0:cutoff_ind])

        # get test performance
        if self.test_labels is not None:
            test_preds = self.ensemble_selection.predict(self.ensemble_predictions_test[0:cutoff_ind])
            if len(test_preds.shape)==3:
                test_preds = test_preds[0]
            if len(test_preds.shape)==2:
                test_preds = np.argmax(test_preds, axis=1)
            test_performance = accuracy(self.test_labels, test_preds)
        else:
            test_performance = 0

        # get ensemble performance on ensemble validation set
        if len(self.labels_ensemble_val)>0 and len(self.ensemble_predictions_ensemble_val)>0:
            ensemble_val_preds = self.ensemble_selection.predict(self.ensemble_predictions_ensemble_val[0:cutoff_ind])
            if len(ensemble_val_preds.shape)==3:
                ensemble_val_preds = ensemble_val_preds[0]
            if len(ensemble_val_preds.shape)==2:
                ensemble_val_preds = np.argmax(ensemble_val_preds, axis=1)
            ensemble_val_performance = accuracy(self.labels_ensemble_val, ensemble_val_preds)
        else:
            ensemble_val_performance = 0

        model_identifiers = self.ensemble_selection.identifiers_
        model_weights = self.ensemble_selection.weights_

        return self.ensemble_selection.get_validation_performance(), test_performance, ensemble_val_performance, model_identifiers, model_weights, ensemble_val_preds, test_preds

    def restart_trajectory_with_reg(self, timelimit=np.inf):
        # For datasets with heavy overfitting reduce considered models
        self.ensemble_config["only_consider_n_best"] = 2
        self.ensemble_selection = EnsembleSelection(**ensemble_config)

        self.simulate_trajectory(timelimit=timelimit, allow_restart=False)

    def simulate_trajectory(self, timelimit=np.inf, allow_restart=False):
        self.trajectory = []
        self.test_trajectory = []
        self.enstest_trajectory = []
        self.model_identifiers = []
        self.model_weights = []
        self.ensemble_loss = []
        self.val_preds = []
        self.test_preds = []

        for ind, t in enumerate(self.timesteps):
            
            if t>timelimit:
                break
            
            print("==> Building ensemble at %i -th timestep %f" %(ind, t))
            ensemble_performance, test_performance, ensemble_val_performance, model_identifiers, model_weights, ensemble_val_preds, test_preds = self.get_ensemble_performance(t)
            print("==> Performance:", ensemble_performance, "/", test_performance, "/", ensemble_val_performance)
            
            if abs(ensemble_performance) == 100 and ind<20 and allow_restart:
                self.restart_trajectory_with_reg(timelimit=np.inf)
                break
            
            self.ensemble_loss.append(ensemble_performance)
            self.trajectory.append((t, ensemble_performance))
            self.test_trajectory.append((t, test_performance))
            self.enstest_trajectory.append((t, ensemble_val_performance))
            self.model_identifiers.append(model_identifiers)
            self.model_weights.append(model_weights)
            self.val_preds.append(ensemble_val_preds)
            self.test_preds.append(test_preds)

    def get_incumbent_at_timestep(self, timestep, use_val=True):
        best_val_score = 0
        best_ind = 0
        if use_val:
            for ind, performance_tuple in enumerate(self.enstest_trajectory):
                if performance_tuple[0]<=timestep and performance_tuple[1]>=best_val_score:
                    best_val_score = performance_tuple[1]
                    best_ind = ind
        else:
            for ind, performance_tuple in enumerate(self.test_trajectory):
                if performance_tuple[0]<=timestep and performance_tuple[1]>=best_val_score:
                    best_val_score = performance_tuple[1]
                    best_ind = ind
        return self.test_trajectory[best_ind], best_ind

    def save_trajectory(self, save_file, test=False):

        print("==> Saving ensemble trajectory to", save_file)

        with open(save_file, "w") as f:
            if test:
                json.dump(self.test_trajectory, f)
            else:
                json.dump(self.trajectory, f)

def get_bohb_rundirs(rundir):

    rundirs = []

    dataset_dirs = [os.path.join(rundir, p) for p in os.listdir(rundir) if not p.endswith("cluster")]

    for ds_path in dataset_dirs:
        rundirs = rundirs + [os.path.join(ds_path, rundir) for rundir in os.listdir(ds_path)]

    return rundirs

def minimize_trf(value):
        return -1*value

def no_transform(value):
    return value

def get_ensemble_config(metric_name="accuracy"):
    autonet_accuracy = AutoNetMetric(name="accuracy", metric=accuracy, loss_transform=minimize_trf, ohe_transform=undo_ohe)
    autonet_cross_entropy = AutoNetMetric(name="cross_entropy", metric=cross_entropy, loss_transform=minimize_trf, ohe_transform=no_transform)
    autonet_auc = AutoNetMetric(name="auc", metric=auc_metric, loss_transform=no_transform, ohe_transform=no_transform)

    METRIC_DICT = {"accuracy": autonet_accuracy,
            "cross_entropy": autonet_cross_entropy,
            "auc_metric": autonet_auc}

    ensemble_config = {"ensemble_size" : 35,
                       "only_consider_n_best" : 10,
                       "sorted_initialization_n_best" : 1,
                       "metric" : METRIC_DICT[metric_name]}
    return ensemble_config


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--rundir", type=str, default="./logs")
    parser.add_argument("--run_id", type=int)
    args = parser.parse_args()

    ensemble_config = get_ensemble_config()

    bohb_rundirs = get_bohb_rundirs(args.rundir)

    print(bohb_rundirs)

    bohb_rundir = bohb_rundirs[args.run_id-1]

    simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=bohb_rundir, ensemble_config=ensemble_config, seed=1)
    simulator.simulate_trajectory()
    simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory.json"))
    simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory_test.json"), test=True)

    #incumbent_score_all_time, incumbent_ind_all_time = simulator.get_incumbent_at_timestep(timestep=np.inf, use_val=False)
    #incumbent_score_all_time_val, incumbent_ind_val_all_time = simulator.get_incumbent_at_timestep(timestep=np.inf, use_val=True)
    incumbent_score_val, incumbent_ind_val = simulator.get_incumbent_at_timestep(timestep=3600, use_val=True)

    incumbent_preds = simulator.test_preds[incumbent_ind_val]
    
    print("Incumbent ind / score:", incumbent_ind_val, "/", incumbent_score_val)

    results = {#"all_time_incumbent":incumbent_score_all_time,
            #"all_time_incumbent_val":incumbent_score_all_time_val,
            #"3600_without_val": score_at_3600,
            "3600_incumbent_val":incumbent_score_val}
            #"3600_incumbent_val":combined_score}

    with open(os.path.join(bohb_rundir, "incumbent_ensemble.json"), "w") as f:
        json.dump(results, f)
