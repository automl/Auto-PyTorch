import os
import json
import time
import argparse
import numpy as np
from IPython import embed

from sklearn.model_selection import KFold
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

    def __init__(self, ensemble_pred_dir, ensemble_config, seed, cross_val, cross_val_split, n_steps=None):
        
        self.ensemble_pred_dir = os.path.join(ensemble_pred_dir, "predictions_for_ensemble.npy")
        self.ensemble_pred_dir_test = os.path.join(ensemble_pred_dir, "test_predictions_for_ensemble.npy")
        self.ensemble_config = ensemble_config
        self.n_steps = n_steps
        self.seed = seed

        self.read_runfiles(cross_val_split=cross_val_split, cross_val=cross_val)
        self.timesteps = self.get_timesteps()
        
        self.ensemble_selection = EnsembleSelection(**ensemble_config)

    def read_runfiles(self, shuffle=False, val_split=0.5, cross_val=3, cross_val_split=1):

        self.ensemble_identifiers = []
        self.ensemble_predictions = []
        self.ensemble_predictions_enstest = []
        self.ensemble_timestamps = []

        with open(self.ensemble_pred_dir, "rb") as f:
            self.labels = np.load(f, allow_pickle=True)
            print("Skipping y transform")

            if shuffle:
                sorting_labels_inv = inv_perm(np.argsort(self.labels))
                full_labels = self.labels

            if val_split is not None and val_split>0:
                # shuffle val data
                indices = np.arange(len(self.labels))
                rng = np.random.default_rng(seed=self.seed)
                rng.shuffle(indices)

                if cross_val>1:
                    kf = KFold(n_splits=cross_val)
                    splits = kf.split(indices)
                    for ind, (train_index, test_index) in enumerate(splits):
                        self.train_indices, self.val_indices = train_index, test_index
                        if ind==cross_val_split:
                            print("TRAIN DATA: ", len(self.train_indices))
                            print("TEST DATA: ", len(self.val_indices))
                            break
                else:
                    split = int(len(indices) * (1-val_split))
                    self.train_indices = indices[:split]
                    self.val_indices = indices[split:]
                
                self.labels_enstest = self.labels[self.val_indices]
                self.labels = self.labels[self.train_indices]

            while True:
                try:
                    job_id, budget, timestamps = np.load(f, allow_pickle=True)
                    predictions = np.array(np.load(f, allow_pickle=True))
                    
                    if shuffle:
                        label_inds = np.arange(len(full_labels))
                        np.random.shuffle(label_inds)
                        labels_temp = full_labels[label_inds]
                        sorting_pred_labels = np.argsort(labels_temp)
                        predictions = predictions[sorting_pred_labels][sorting_labels_inv]
                    
                    if val_split is not None and val_split>0:
                        self.ensemble_identifiers.append(job_id + (budget, ))
                        self.ensemble_predictions.append(predictions[self.train_indices])
                        self.ensemble_predictions_enstest.append(predictions[self.val_indices])
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
                    print("Skipping y transform")
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

        self.transform_timestamps(add_time=-self.ensemble_timestamps[0]["submitted"])

        print("==> Found %i val preds" %len(self.ensemble_predictions))
        print("==> Found %i test preds" %len(self.ensemble_predictions_test))
        print("==> Found %i timestamps" %len(self.ensemble_timestamps))

    def transform_timestamps(self, add_time):
        # timestamps are sorted
        transformed_timestamps = [t["finished"]+add_time for t in self.ensemble_timestamps]
        self.ensemble_timestamps = transformed_timestamps

    def get_timesteps(self):
        # we want at least 2 models
        first_timestep = self.ensemble_timestamps[1]
        final_timestep = self.ensemble_timestamps[-1]
        if self.n_steps is not None:
            return np.linspace(first_timestep, final_timestep, self.n_steps)
        return self.ensemble_timestamps[1:]

    def get_ensemble_performance(self, timestep):
        cutoff_ind = np.argmin([abs(t - timestep) for t in self.ensemble_timestamps])+1
        print("==> Considering %i models and timestep %f" %(cutoff_ind, timestep))

        self.ensemble_selection.fit(np.array(self.ensemble_predictions[0:cutoff_ind]), self.labels, self.ensemble_identifiers[0:cutoff_ind])

        #print("==> Ensemble weights (%i):" %len(self.ensemble_selection.weights_), self.ensemble_selection.weights_)

        if self.test_labels is not None:
            #print("==> Test preds: %i" %len(self.ensemble_predictions_test))
            test_preds = self.ensemble_selection.predict(self.ensemble_predictions_test[0:cutoff_ind])
            if len(test_preds.shape)==3:
                #embed()
                test_preds = test_preds[0]
            if len(test_preds.shape)==2:
                test_preds = np.argmax(test_preds, axis=1)
            test_performance = accuracy(self.test_labels, test_preds)
        else:
            test_performance = 0

        if len(self.ensemble_pred_dir_test)>0:
            enstest_preds = self.ensemble_selection.predict(self.ensemble_predictions_enstest[0:cutoff_ind])
            if len(enstest_preds.shape)==3:
                enstest_preds = enstest_preds[0]
            if len(enstest_preds.shape)==2:
                enstest_preds = np.argmax(enstest_preds, axis=1)
            enstest_performance = accuracy(self.labels_enstest, enstest_preds)
        else:
            enstest_performance = 0

        model_identifiers = self.ensemble_selection.identifiers_
        model_weights = self.ensemble_selection.weights_

        return self.ensemble_selection.get_validation_performance(), test_performance, enstest_performance, model_identifiers, model_weights

    def simulate_trajectory(self):
        self.trajectory = []
        self.test_trajectory = []
        self.enstest_trajectory = []
        self.model_identifiers = []
        self.model_weights = []
        for ind, t in enumerate(self.timesteps):
            print("==> Building ensemble at %i -th timestep %f" %(ind, t))
            ensemble_performance, test_performance, enstest_performance, model_identifiers, model_weights = self.get_ensemble_performance(t)
            print("==> Performance:", ensemble_performance, "/", test_performance, "/", enstest_performance)
            self.trajectory.append((t, ensemble_performance))
            self.test_trajectory.append((t, test_performance))
            self.enstest_trajectory.append((t, enstest_performance))
            self.model_identifiers.append(model_identifiers)
            self.model_weights.append(model_weights)
            #print(self.trajectory[-1])

    def predict_with_weights(self, identifiers, weights):
        self.ensemble_selection.identifiers_ = identifiers
        self.ensemble_selection.weights_ = weights
        
        print(len(identifiers))
        print(len(weights))
        print(len(self.ensemble_predictions_test))
        print(len(self.test_labels))

        test_preds = self.ensemble_selection.predict(self.ensemble_predictions_test[0:len(identifiers)])
        if len(test_preds.shape)==3:
            test_preds = test_preds[0]
        if len(test_preds.shape)==2:
            test_preds = np.argmax(test_preds, axis=1)
        test_performance = accuracy(self.test_labels, test_preds)
        return test_performance

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

    def get_score_at_timestep(self, timestep):
        for ind, performance_tuple in enumerate(self.test_trajectory):
            if performance_tuple[0]>timestep:
                break
        return self.test_trajectory[ind-1], ind

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

    dataset_dirs = [os.path.join(rundir, p) for p in os.listdir(rundir) if not p.endswith("cluster")]

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

    ensemble_config = {"ensemble_size" : 35, #35
                       "only_consider_n_best" : 10, #10
                       "sorted_initialization_n_best" : 1,
                       #"only_consider_n_best_percent" : 0,
                       "metric" : autonet_accuracy}

    if args.test=="true":
        from sklearn import metrics
        test_dir = args.rundir
        test_dir = "/home/zimmerl/Auto-PyTorch_releases/Auto-PyTorch/logs/4/run_5"

        #simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=test_dir, ensemble_config=ensemble_config, n_steps=3)
        simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=test_dir, ensemble_config=ensemble_config)
        simulator.simulate_trajectory()
        simulator.print_pred_heads()
        print("==> Val trajectory:", simulator.trajectory)
        print("==> Test trajectory:", simulator.test_trajectory)
        #embed()
        raise NotImplementedError


    # Get
    bohb_rundirs = get_bohb_rundirs(args.rundir)

    print(bohb_rundirs)

    bohb_rundir = bohb_rundirs[args.run_id-1]

    # Vanilla APT
    num_cross_val_splits = 1
    incumbent_identifiers = []
    incumbent_weights = []
    identifier_weight_dicts = []
    for ind in range(num_cross_val_splits): # cross_val #NOTE: proper cross val tbd, this is just seeds
        simulator = EnsembleTrajectorySimulator(ensemble_pred_dir=bohb_rundir, ensemble_config=ensemble_config, seed=ind, cross_val=1, cross_val_split=1)
        simulator.simulate_trajectory()
        simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory.json"))
        simulator.save_trajectory(save_file=os.path.join(bohb_rundir, "ensemble_trajectory_test.json"), test=True)

        incumbent_score_all_time, _ = simulator.get_incumbent_at_timestep(timestep=np.inf, use_val=False)
        incumbent_score_all_time_val, _ = simulator.get_incumbent_at_timestep(timestep=np.inf, use_val=True)
        incumbent_score_val, incumbent_ind_val = simulator.get_incumbent_at_timestep(timestep=3600, use_val=True)
        
        score_at_3600, _ = simulator.get_score_at_timestep(timestep=3600)

        identifiers = simulator.model_identifiers[incumbent_ind_val]
        weights = simulator.model_weights[incumbent_ind_val]
        
        incumbent_identifiers.append(identifiers)
        incumbent_weights.append(weights)
        
        ident_weight_dict = dict()
        for ident, weight in zip(identifiers, weights):
            ident_weight_dict[ident] = weight
        identifier_weight_dicts.append(ident_weight_dict)
        
    all_identifiers = simulator.model_identifiers[-1]
    weight_dict = {ident:0 for ident in all_identifiers}

    for ident_weight_dict in identifier_weight_dicts:
        for key, val in ident_weight_dict.items():
            weight_dict[key] += val/num_cross_val_splits

    combined_score = simulator.predict_with_weights(list(ident_weight_dict.keys()), list(ident_weight_dict.values()))

    print(identifier_weight_dicts)

    print(weight_dict)

    print("Combined score / true score:", combined_score, "/", incumbent_score_val)

    results = {"all_time_incumbent":incumbent_score_all_time,
            "all_time_incumbent_val":incumbent_score_all_time_val,
            "3600_without_val": score_at_3600,
            "3600_incumbent_val":incumbent_score_val}
            #"3600_incumbent_val":combined_score}

    with open(os.path.join(bohb_rundir, "incumbent_ensemble.json"), "w") as f:
        json.dump(results, f)
