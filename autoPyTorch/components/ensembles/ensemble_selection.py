from collections import Counter
import random

import numpy as np

from autoPyTorch.components.ensembles.abstract_ensemble import AbstractEnsemble


class EnsembleSelection(AbstractEnsemble):
    """Ensemble Selection algorithm extracted from auto-sklearn"""
    
    def __init__(self, ensemble_size, metric,
                 sorted_initialization_n_best=0, only_consider_n_best=0,
                 bagging=False, mode='fast'):
        self.ensemble_size = ensemble_size
        self.metric = metric.get_loss_value
        self.sorted_initialization_n_best = sorted_initialization_n_best
        self.only_consider_n_best = only_consider_n_best
        self.bagging = bagging
        self.mode = mode

    def fit(self, predictions, labels, identifiers):
        self.ensemble_size = int(self.ensemble_size)
        if self.ensemble_size < 1:
            raise ValueError('Ensemble size cannot be less than one!')
        if self.mode not in ('fast', 'slow'):
            raise ValueError('Unknown mode %s' % self.mode)

        if self.bagging:
            self._bagging(predictions, labels)
        else:
            self._fit(predictions, labels)
        self._calculate_weights()
        self.identifiers_ = identifiers
        return self

    def _fit(self, predictions, labels):
        if self.mode == 'fast':
            self._fast(predictions, labels)
        else:
            self._slow(predictions, labels)
        return self

    def _fast(self, predictions, labels):
        """Fast version of Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization_n_best > 0:
            indices = self._sorted_initialization(predictions, labels, self.sorted_initialization_n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = self.metric(ensemble_, labels)
                trajectory.append(ensemble_performance)
            ensemble_size -= self.sorted_initialization_n_best
        
        only_consider_indices = None
        if self.only_consider_n_best > 0:
            only_consider_indices = set(self._sorted_initialization(predictions, labels, self.only_consider_n_best))

        for i in range(ensemble_size):
            scores = np.zeros((len(predictions)))
            s = len(ensemble)
            if s == 0:
                weighted_ensemble_prediction = np.zeros(predictions[0].shape)
            else:
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                weighted_ensemble_prediction = (s / float(s + 1)) * \
                                               ensemble_prediction
            fant_ensemble_prediction = np.zeros(weighted_ensemble_prediction.shape)
            for j, pred in enumerate(predictions):
                # TODO: this could potentially be vectorized! - let's profile
                # the script first!
                if only_consider_indices and j not in only_consider_indices:
                    scores[j] = float("inf")
                    continue
                fant_ensemble_prediction[:,:] = weighted_ensemble_prediction + \
                                             (1. / float(s + 1)) * pred
                scores[j] = self.metric(fant_ensemble_prediction, labels)
            all_best = np.argwhere(scores == np.nanmin(scores)).flatten()
            best = np.random.choice(all_best)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = order
        self.trajectory_ = trajectory
        self.train_score_ = trajectory[-1]

    def _slow(self, predictions, labels):
        """Rich Caruana's ensemble selection method."""
        self.num_input_models_ = len(predictions)

        ensemble = []
        trajectory = []
        order = []

        ensemble_size = self.ensemble_size

        if self.sorted_initialization_n_best > 0:
            indices = self._sorted_initialization(predictions, labels, self.sorted_initialization_n_best)
            for idx in indices:
                ensemble.append(predictions[idx])
                order.append(idx)
                ensemble_ = np.array(ensemble).mean(axis=0)
                ensemble_performance = self.metric(ensemble_, labels)
                trajectory.append(ensemble_performance)
            ensemble_size -= self.sorted_initialization_n_best
        
        only_consider_indices = None
        if self.only_consider_n_best > 0:
            only_consider_indices = set(self._sorted_initialization(predictions, labels, self.only_consider_n_best))

        for i in range(ensemble_size):
            scores = np.zeros([predictions.shape[0]])
            for j, pred in enumerate(predictions):
                if only_consider_indices and j not in only_consider_indices:
                    scores[j] = float("inf")
                    continue
                ensemble.append(pred)
                ensemble_prediction = np.mean(np.array(ensemble), axis=0)
                scores[j] = self.metric(ensemble_prediction, labels)
                ensemble.pop()
            best = np.nanargmin(scores)
            ensemble.append(predictions[best])
            trajectory.append(scores[best])
            order.append(best)

            # Handle special case
            if len(predictions) == 1:
                break

        self.indices_ = np.array(order)
        self.trajectory_ = np.array(trajectory)
        self.train_score_ = trajectory[-1]

    def _calculate_weights(self):
        ensemble_members = Counter(self.indices_).most_common()
        weights = np.zeros((self.num_input_models_,), dtype=float)
        for ensemble_member in ensemble_members:
            weight = float(ensemble_member[1]) / self.ensemble_size
            weights[ensemble_member[0]] = weight

        if np.sum(weights) < 1:
            weights = weights / np.sum(weights)

        self.weights_ = weights

    def _sorted_initialization(self, predictions, labels, n_best):
        perf = np.zeros([predictions.shape[0]])

        for idx, prediction in enumerate(predictions):
            perf[idx] = self.metric(prediction, labels)

        indices = np.argsort(perf)[:n_best]
        return indices

    def _bagging(self, predictions, labels, fraction=0.5, n_bags=20):
        """Rich Caruana's ensemble selection method with bagging."""
        raise ValueError('Bagging might not work with class-based interface!')
        n_models = predictions.shape[0]
        bag_size = int(n_models * fraction)

        for j in range(n_bags):
            # Bagging a set of models
            indices = sorted(random.sample(range(0, n_models), bag_size))
            bag = predictions[indices, :, :]
            self._fit(bag, labels)

    def predict(self, predictions):
        if len(predictions) < len(self.weights_):
            weights = (weight for  weight in self.weights_ if weight > 0)
        else:
            weights = self.weights_

        for i, weight in enumerate(weights):
            predictions[i] *= weight
        return np.sum(predictions, axis=0)

    def __str__(self):
        return 'Ensemble Selection:\n\tTrajectory: %s\n\tMembers: %s' \
               '\n\tWeights: %s\n\tIdentifiers: %s' % \
               (' '.join(['%d: %5f' % (idx, performance)
                         for idx, performance in enumerate(self.trajectory_)]),
                self.indices_, self.weights_,
                ' '.join([str(identifier) for idx, identifier in
                          enumerate(self.identifiers_)
                          if self.weights_[idx] > 0]))

    def get_models_with_weights(self, models):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                model = models[identifier]
                output.append((weight, model))

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_selected_model_identifiers(self):
        output = []

        for i, weight in enumerate(self.weights_):
            identifier = self.identifiers_[i]
            if weight > 0.0:
                output.append(identifier)

        output.sort(reverse=True, key=lambda t: t[0])

        return output

    def get_validation_performance(self):
        return self.trajectory_[-1]
