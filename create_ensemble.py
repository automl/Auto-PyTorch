import os
import json
import time
import argparse
import numpy as np
from IPython import embed

from autoPyTorch.components.metrics import accuracy
from autoPyTorch.pipeline.nodes.metric_selector import AutoNetMetric, undo_ohe, default_minimize_transform
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection
from hpbandster.core.result import logged_results_to_HBS_result


def inv_perm(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse


class EnsembleTrajectorySimulator():

    def __init__(self, ensemble_pred_dir, ensemble_config, n_steps=None):
        
        self.ensemble_pred_dir = os.path.join(ensemble_pred_dir, "predictions_for_ensemble.npy")
        self.ensemble_pred_dir_test = os.path.join(ensemble_pred_dir, "test_predictions_for_ensemble.npy")
        self.ensemble_config = ensemble_config
        self.n_steps = n_steps

        self.read_runfiles()
        self.timesteps = self.get_timesteps()

        self.ensemble_selection = EnsembleSelection(**ensemble_config)

    def read_runfiles(self):

        self.ensemble_identifiers = []
        self.ensemble_predictions = []
        self.ensemble_timestamps = []

        with open(self.ensemble_pred_dir, "rb") as f:
            self.labels = np.load(f, allow_pickle=True)
            print("Skipping y transform")

            while True:
                try:
                    job_id, budget, timestamps = np.load(f, allow_pickle=True)
                    predictions = np.array(np.load(f, allow_pickle=True))
                    
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
                    print("Skipping y transform")
                except (EOFError, OSError):
                    pass

                while True:
                    try:
                        job_id, budget, timestamps = np.load(f, allow_pickle=True)
                        predictions = np.array(np.load(f, allow_pickle=True))

                        self.ensemble_predictions_test.append(predictions)
                        print("==> Adding test labels with shape", predictions.shape)
                    except (EOFError, OSError):
                        break

        self.transform_timestamps(add_time=-self.ensemble_timestamps[0]["submitted"])

        print("==> Found %i val preds" %len(self.ensemble_predictions))
        print("==> Found %i test preds" %len(self.ensemble_predictions_test))
        print("==> Found %i timestamps" %len(self.ensemble_timestamps))

    def transform_timestamps(self, add_time):
        #NOTE: timestamps are sorted
        transformed_timestamps = [t["finished"]+add_time for t in self.ensemble_timestamps]
        self.ensemble_timestamps = transformed_timestamps

    def get_timesteps(self):
        #NOTE: we want at least 2 models
        first_timestep = self.ensemble_timestamps[1]
        final_timestep = self.ensemble_timestamps[-1]
        if self.n_steps is not None:
            return np.linspace(first_timestep, final_timestep, self.n_steps)
        return self.ensemble_timestamps[1:]

    def get_ensemble_performance(self, timestep):
        cutoff_ind = np.argmin([abs(t - timestep) for t in self.ensemble_timestamps])+1   #MODIFIED
        print("==> Considering %i models and timestep %f" %(cutoff_ind, timestep))

        self.ensemble_selection.fit(np.array(self.ensemble_predictions[0:cutoff_ind]), self.labels, self.ensemble_identifiers[0:cutoff_ind])

        print("==> Ensemble weights (%i):" %len(self.ensemble_selection.weights_), self.ensemble_selection.weights_)

        if self.test_labels is not None:
            print("==> Test preds: %i" %len(self.ensemble_predictions_test))
            test_preds = self.ensemble_selection.predict(self.ensemble_predictions_test[0:cutoff_ind])
            if len(test_preds.shape)==3:
                test_preds = test_preds[0]
            if len(test_preds.shape)==2:
                test_preds = np.argmax(test_preds, axis=1)
            test_performance = accuracy(self.test_labels, test_preds)
        else:
            test_performance = 0

        return self.ensemble_selection.get_validation_performance(), test_performance

    def simulate_trajectory(self):
        self.trajectory = []
        self.test_trajectory = []
        for ind, t in enumerate(self.timesteps):
            print("==> Building ensemble at %i -th timestep %f" %(ind, t))
            ensemble_performance, test_performance = self.get_ensemble_performance(t)
            print("==> Performance:", ensemble_performance, "/", test_performance)
            self.trajectory.append((t, ensemble_performance))
            self.test_trajectory.append((t, test_performance))
            #print(self.trajectory[-1])

    def save_trajectory(self, save_file, test=False):

        print("==> Saving ensemble trajectory to", save_file)

        with open(save_file, "w") as f:
            if test:
                json.dump(self.test_trajectory, f)
            else:
                json.dump(self.trajectory, f)

    def print_pred_heads(self):
        heads = [p[0:3] for p in self.ensemble_predictions[0:5]]
        print("\nPred 1:", heads[0][0])
        print("Pred 2:", heads[1][0])
        print("Pred 3:", heads[2][0], "\n")

def get_bohb_rundirs(rundir):

    rundirs = []

    dataset_dirs = [os.path.join(rundir, p) for p in os.listdir(rundir)]

    for ds_path in dataset_dirs:
        rundirs = rundirs + [os.path.join(ds_path, rundir) for rundir in os.listdir(ds_path)]

    return rundirs


if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--rundir", type=str, default="/home/zimmerl/Auto-PyTorch_releases/Auto-PyTorch/logs")
    parser.add_argument("--run_id", type=int)
    parser.add_argument("--test", type=str, default="false")
    args = parser.parse_args()

    def minimize_trf(value):
        return -1*value

    autonet_accuracy = AutoNetMetric(name="accuracy", metric=accuracy, loss_transform=minimize_trf, ohe_transform=undo_ohe)

    ensemble_config = {"ensemble_size" : 50,
                       "only_consider_n_best" : 30,
                       "sorted_initialization_n_best" : 0,
                       #"only_consider_n_best_percent" : 0,
                       "metric" : autonet_accuracy}

    if args.test=="true":
        from sklearn import metrics
        test_dir = args.rundir
        test_dir = "/home/zimmerl/Auto-PyTorch_releases/Auto-PyTorch/logs/0/run_1"

        #simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=test_dir, ensemble_config=ensemble_config, n_steps=3)
        simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=test_dir, ensemble_config=ensemble_config)
        simulator.simulate_trajectory()
        simulator.print_pred_heads()
        print("==> Val trajectory:", simulator.trajectory)
        print("==> Test trajectory:", simulator.test_trajectory)
        embed()
        raise NotImplementedError


    # Get
    bohb_rundirs = get_bohb_rundirs(args.rundir)

    print(bohb_rundirs)

    bohb_rundir = bohb_rundirs[args.run_id-1]

    print(bohb_rundir, dataset)

    # Vanilla APT
    simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=bohb_rundir, ensemble_config=ensemble_config)
    simulator.simulate_trajectory()
    simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory.json"))
    simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory_test.json"), test=True)
