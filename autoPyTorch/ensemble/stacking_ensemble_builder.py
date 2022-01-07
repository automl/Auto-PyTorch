import glob
import logging
import logging.handlers
import math
from mmap import MADV_NOHUGEPAGE
import os
import pickle
import re
import time
import traceback
import zlib
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.random.mtrand import seed

from autoPyTorch.automl_common.common.utils.backend import Backend
from autoPyTorch.constants import BINARY
from autoPyTorch.ensemble.abstract_ensemble import AbstractEnsemble
from autoPyTorch.ensemble.ensemble_builder import EnsembleBuilder
from autoPyTorch.ensemble.stacking_ensemble import StackingEnsemble
from autoPyTorch.pipeline.components.training.metrics.base import autoPyTorchMetric
from autoPyTorch.pipeline.components.training.metrics.utils import calculate_loss
from autoPyTorch.utils.logging_ import get_named_client_logger

Y_ENSEMBLE = 0
Y_TEST = 1

MODEL_FN_RE = r'_([0-9]*)_([0-9]*)_([0-9]+\.*[0-9]*)\.npy'


# TODO: think of what functions are needed to support stacking
# TODO: make functions to support stacking.
class StackingEnsembleBuilder(EnsembleBuilder):
    def __init__(
        self,
        backend: Backend,
        dataset_name: str,
        task_type: int,
        output_type: int,
        metrics: List[autoPyTorchMetric],
        opt_metric: str,
        ensemble_size: int = 10,
        ensemble_nbest: int = 100,
        max_models_on_disc: Union[float, int] = 100,
        performance_range_threshold: float = 0,
        seed: int = 1,
        precision: int = 32,
        memory_limit: Optional[int] = 1024,
        read_at_most: int = 5,
        random_state: Optional[Union[int, np.random.RandomState]] = None,
        logger_port: int = logging.handlers.DEFAULT_TCP_LOGGING_PORT,
        unit_test: bool = False,
    ):
        """
            Constructor
            Parameters
            ----------
            backend: util.backend.Backend
                backend to write and read files
            dataset_name: str
                name of dataset
            task_type: int
                type of ML task
            metrics: List[autoPyTorchMetric],
                name of metric to score predictions
            opt_metric: str
                name of the metric to optimize
            ensemble_size: int
                maximal size of ensemble (passed to ensemble.ensemble_selection)
            ensemble_nbest: int/float
                if int: consider only the n best prediction
                if float: consider only this fraction of the best models
                Both wrt to validation predictions
                If performance_range_threshold > 0, might return less models
            max_models_on_disc: Union[float, int]
               Defines the maximum number of models that are kept in the disc.
               If int, it must be greater or equal than 1, and dictates the max number of
               models to keep.
               If float, it will be interpreted as the max megabytes allowed of disc space. That
               is, if the number of ensemble candidates require more disc space than this float
               value, the worst models will be deleted to keep within this budget.
               Models and predictions of the worst-performing models will be deleted then.
               If None, the feature is disabled.
               It defines an upper bound on the models that can be used in the ensemble.
            performance_range_threshold: float
                Keep only models that are better than:
                    dummy + (best - dummy)*performance_range_threshold
                E.g dummy=2, best=4, thresh=0.5 --> only consider models with score > 3
                Will at most return the minimum between ensemble_nbest models,
                and max_models_on_disc. Might return less
            seed: int
                random seed
            precision: [16,32,64,128]
                precision of floats to read the predictions
            memory_limit: Optional[int]
                memory limit in mb. If ``None``, no memory limit is enforced.
            read_at_most: int
                read at most n new prediction files in each iteration
            logger_port: int
                port that receives logging records
            unit_test: bool
                Turn on unit testing mode. This currently makes fit_ensemble raise a MemoryError.
                Having this is very bad coding style, but I did not find a way to make
                unittest.mock work through the pynisher with all spawn contexts. If you know a
                better solution, please let us know by opening an issue.
        """

        super(StackingEnsembleBuilder, self).__init__(
            backend=backend, dataset_name=dataset_name, task_type=task_type,
            output_type=output_type, metrics=metrics, opt_metric=opt_metric,
            ensemble_size=ensemble_size, ensemble_nbest=ensemble_nbest,
            max_models_on_disc=max_models_on_disc,
            performance_range_threshold=performance_range_threshold,
            seed=seed, precision=precision, memory_limit=memory_limit,
            read_at_most=read_at_most, random_state=random_state,
            logger_port=logger_port, unit_test=unit_test)

    # TODO: This is the main wrapper to the EnsembleSelection class which fits
    # TODO: the ensemble
    def main(
        self, time_left: float, iteration: int, return_predictions: bool,
    ) -> Tuple[
        List[Dict[str, float]],
        int,
        Optional[np.ndarray],
        Optional[np.ndarray],
    ]:
        """
        This is the main function of the ensemble builder process and can be considered
        a wrapper over the ensemble selection method implemented y EnsembleSelection class.

        This method is going to be called multiple times by the main process, to
        build and ensemble, in case the SMAC process produced new models and to provide
        anytime results.

        On this regard, this method mainly:
            1- select from all the individual models that smac created, the N-best candidates
               (this in the scenario that N > ensemble_nbest argument to this class). This is
               done based on a score calculated via the metrics argument.
            2- This pre-selected candidates are provided to the ensemble selection method
               and if a ensemble is found under the provided memory/time constraints, a new
               ensemble is proposed.
            3- Because this process will be called multiple times, it performs checks to make
               sure a new ensenmble is only proposed if new predictions are available, as well
               as making sure we do not run out of resources (like disk space)

        Args:
            time_left (float):
                How much time is left for the ensemble builder process
            iteration (int):
                Which is the current iteration
            return_predictions (bool):
                Whether we want to return the predictions of the current model or not

        Returns:
            ensemble_history (Dict):
                A snapshot of both test and optimization performance. For debugging.
            ensemble_nbest (int):
                The user provides a direction on how many models to use in ensemble selection.
                This number can be reduced internally if the memory requirements force it.
            train_predictions (np.ndarray):
                The optimization prediction from the current ensemble.
            test_predictions (np.ndarray):
                The train prediction from the current ensemble.
        """

        # Pynisher jobs inside dask 'forget'
        # the logger configuration. So we have to set it up
        # accordingly
        self.logger = get_named_client_logger(
            name='EnsembleBuilder',
            port=self.logger_port,
        )

        self.start_time = time.time()
        train_pred, test_pred = None, None

        used_time = time.time() - self.start_time
        self.logger.debug(
            'Starting iteration %d, time left: %f',
            iteration,
            time_left - used_time,
        )

        ensemble_identifiers = None
        # Get ensemble_identifiers from previous iteration.
        ensemble_dir = self.backend.get_ensemble_dir()
        if len(os.listdir(ensemble_dir)) >= 1:
            old_ensemble = self.backend.load_ensemble(self.seed)
            ensemble_identifiers = old_ensemble.ensemble_identifiers

        self.ensemble_slot_j = np.mod(iteration, self.ensemble_size)

        # populates self.read_preds and self.read_losses with individual model predictions and ensemble loss.
        if not self.compute_ensemble_loss_per_model(ensemble_identifiers=ensemble_identifiers):
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # Only the models with the n_best predictions are candidates
        # to be in the ensemble
        candidate_models = self.get_n_best_preds()
        if not candidate_models:  # no candidates yet
            if return_predictions:
                return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
            else:
                return self.ensemble_history, self.ensemble_nbest, None, None

        # populates test predictions in self.read_preds
        # reduces selected models if file reading failed
        n_sel_test = self.get_test_preds(selected_keys=candidate_models)

        # If any of n_sel_* is not empty and overlaps with candidate_models,
        # then ensure candidate_models AND n_sel_test are sorted the same
        candidate_models_set = set(candidate_models)
        if candidate_models_set.intersection(n_sel_test):
            candidate_models = sorted(list(candidate_models_set.intersection(
                n_sel_test)))
            n_sel_test = candidate_models
        else:
            # This has to be the case
            n_sel_test = []

        if os.environ.get('ENSEMBLE_KEEP_ALL_CANDIDATES'):
            for candidate in candidate_models:
                self._has_been_candidate.add(candidate)

        # as candidate models is sorted in `get_n_best_preds`
        best_model_identifier = candidate_models[0]

        # initialise ensemble_identifier with best_model_identifier
        if ensemble_identifiers == None:
            ensemble_identifiers = [best_model_identifier]

        # train ensemble
        ensemble = self.fit_ensemble(
            selected_keys=candidate_models,
            ensemble_identifiers=ensemble_identifiers,
            best_model_identifier=best_model_identifier
            )

        # Save the ensemble for later use in the main module!
        if ensemble is not None and self.SAVE2DISC:
            self.backend.save_ensemble(ensemble, iteration, self.seed)

        # Delete files of non-candidate models - can only be done after fitting the ensemble and
        # saving it to disc so we do not accidentally delete models in the previous ensemble
        if self.max_resident_models is not None:
            self._delete_excess_models(selected_keys=candidate_models)

        # Save the read losses status for the next iteration
        with open(self.ensemble_loss_file, "wb") as memory:
            pickle.dump(self.read_losses, memory)

        if ensemble is not None:
            train_pred = self.predict(set_="train",
                                      ensemble=ensemble,
                                      selected_keys=candidate_models,
                                      n_preds=len(candidate_models),
                                      index_run=iteration)
            # TODO if predictions fails, build the model again during the
            #  next iteration!
            test_pred = self.predict(set_="test",
                                     ensemble=ensemble,
                                     selected_keys=n_sel_test,
                                     n_preds=len(candidate_models),
                                     index_run=iteration)

            # Add a score to run history to see ensemble progress
            self._add_ensemble_trajectory(
                train_pred,
                test_pred
            )

        # The loaded predictions and the hash can only be saved after the ensemble has been
        # built, because the hash is computed during the construction of the ensemble
        with open(self.ensemble_memory_file, "wb") as memory:
            pickle.dump((self.read_preds, self.last_hash), memory)

        if return_predictions:
            return self.ensemble_history, self.ensemble_nbest, train_pred, test_pred
        else:
            return self.ensemble_history, self.ensemble_nbest, None, None

    # TODO: change this function, to compute loss according to Lavesque et al.
    # TODO: this will help us in choosing the model with the lowest ensemble error.
    # TODO: predictions on ensemble set will be available in read_preds to be used for
    # TODO: passing to stacking_ensemble_builder.predict()
    def compute_ensemble_loss_per_model(self, ensemble_identifiers) -> bool:
        """
            Compute the loss of the predictions on ensemble building data set;
            populates self.read_preds and self.read_losses
        """

        self.logger.debug("Read ensemble data set predictions")

        if self.y_true_ensemble is None:
            try:
                self.y_true_ensemble = self.backend.load_targets_ensemble()
            except FileNotFoundError:
                self.logger.debug(
                    "Could not find true targets on ensemble data set: %s",
                    traceback.format_exc(),
                )
                return False

        pred_path = os.path.join(
            glob.escape(self.backend.get_runs_directory()),
            '%d_*_*' % self.seed,
            'predictions_ensemble_%s_*_*.npy*' % self.seed,
        )
        y_ens_files = glob.glob(pred_path)
        y_ens_files = [y_ens_file for y_ens_file in y_ens_files
                       if y_ens_file.endswith('.npy') or y_ens_file.endswith('.npy.gz')]
        self.y_ens_files = y_ens_files
        # no validation predictions so far -- no files
        if len(self.y_ens_files) == 0:
            self.logger.debug("Found no prediction files on ensemble data set:"
                              " %s" % pred_path)
            return False

        # First sort files chronologically
        to_read = []
        for y_ens_fn in self.y_ens_files:
            match = self.model_fn_re.search(y_ens_fn)
            if match is None:
                raise ValueError(f"Could not interpret file {y_ens_fn} "
                                 "Something went wrong while scoring predictions")
            _seed = int(match.group(1))
            _num_run = int(match.group(2))
            _budget = float(match.group(3))

            to_read.append([y_ens_fn, match, _seed, _num_run, _budget])

        n_read_files = 0
        # Now read file wrt to num_run
        # Mypy assumes sorted returns an object because of the lambda. Can't get to recognize the list
        # as a returning list, so as a work-around we skip next line
        for y_ens_fn, match, _seed, _num_run, _budget in sorted(to_read, key=lambda x: x[3]):  # type: ignore
            if self.read_at_most and n_read_files >= self.read_at_most:
                # limit the number of files that will be read
                # to limit memory consumption
                break

            if not y_ens_fn.endswith(".npy") and not y_ens_fn.endswith(".npy.gz"):
                self.logger.info('Error loading file (not .npy or .npy.gz): %s', y_ens_fn)
                continue

            if not self.read_losses.get(y_ens_fn):
                self.read_losses[y_ens_fn] = {
                    "ens_loss": np.inf,
                    "mtime_ens": 0,
                    "mtime_test": 0,
                    "seed": _seed,
                    "num_run": _num_run,
                    "budget": _budget,
                    "disc_space_cost_mb": None,
                    # Lazy keys so far:
                    # 0 - not loaded
                    # 1 - loaded and in memory
                    # 2 - loaded but dropped again
                    # 3 - deleted from disk due to space constraints
                    "loaded": 0
                }
            if not self.read_preds.get(y_ens_fn):
                self.read_preds[y_ens_fn] = {
                    Y_ENSEMBLE: None,
                    Y_TEST: None,
                }

            if self.read_losses[y_ens_fn]["mtime_ens"] == os.path.getmtime(y_ens_fn):
                # same time stamp; nothing changed;
                continue

            # actually read the predictions and compute their respective loss
            try:
                y_ensemble = self._read_np_fn(y_ens_fn)
                losses = self.get_ensemble_loss(
                    ensemble_identifiers=ensemble_identifiers,
                    model_predictions=y_ensemble
                    )

                if np.isfinite(self.read_losses[y_ens_fn]["ens_loss"]):
                    self.logger.debug(
                        'Changing ensemble loss for file %s from %f to %f '
                        'because file modification time changed? %f - %f',
                        y_ens_fn,
                        self.read_losses[y_ens_fn]["ens_loss"],
                        losses[self.opt_metric],
                        self.read_losses[y_ens_fn]["mtime_ens"],
                        os.path.getmtime(y_ens_fn),
                    )

                self.read_losses[y_ens_fn]["ens_loss"] = losses[self.opt_metric]

                # It is not needed to create the object here
                # To save memory, we just compute the loss.
                self.read_losses[y_ens_fn]["mtime_ens"] = os.path.getmtime(y_ens_fn)
                self.read_losses[y_ens_fn]["loaded"] = 2
                self.read_losses[y_ens_fn]["disc_space_cost_mb"] = self.get_disk_consumption(
                    y_ens_fn
                )

                n_read_files += 1

            except Exception:
                self.logger.warning(
                    'Error loading %s: %s',
                    y_ens_fn,
                    traceback.format_exc(),
                )
                self.read_losses[y_ens_fn]["ens_loss"] = np.inf

        self.logger.debug(
            'Done reading %d new prediction files. Loaded %d predictions in '
            'total.',
            n_read_files,
            np.sum([pred["loaded"] > 0 for pred in self.read_losses.values()])
        )
        return True

    def get_test_preds(self, selected_keys: List[str]) -> List[str]:
        """
        test predictions from disc
        and store them in self.read_preds
        Parameters
        ---------
        selected_keys: list
            list of selected keys of self.read_preds
        Return
        ------
        success_keys:
            all keys in selected keys for which we could read the valid and
            test predictions
        """
        success_keys_test = []

        for k in selected_keys:
            test_fn = glob.glob(
                os.path.join(
                    glob.escape(self.backend.get_runs_directory()),
                    '%d_%d_%s' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"],
                    ),
                    'predictions_test_%d_%d_%s.npy*' % (
                        self.read_losses[k]["seed"],
                        self.read_losses[k]["num_run"],
                        self.read_losses[k]["budget"]
                    )
                )
            )
            test_fn = [tfn for tfn in test_fn if tfn.endswith('.npy') or tfn.endswith('.npy.gz')]

            if len(test_fn) == 0:
                # self.logger.debug("Not found test prediction file (although "
                #                   "ensemble predictions available):%s" %
                #                   test_fn)
                pass
            else:
                if (
                    self.read_losses[k]["mtime_test"] == os.path.getmtime(test_fn[0])
                    and k in self.read_preds
                    and self.read_preds[k][Y_TEST] is not None
                ):
                    success_keys_test.append(k)
                    continue
                try:
                    y_test = self._read_np_fn(test_fn[0])
                    self.read_preds[k][Y_TEST] = y_test
                    success_keys_test.append(k)
                    self.read_losses[k]["mtime_test"] = os.path.getmtime(test_fn[0])
                except Exception:
                    self.logger.warning('Error loading %s: %s',
                                        test_fn, traceback.format_exc())

        return success_keys_test

    def fit_ensemble(
        self,
        selected_keys: List[str],
        best_model_identifier,
        ensemble_identifiers = None
    ) -> Optional[StackingEnsemble]:
        """
            fit ensemble

            Parameters
            ---------
            selected_keys: list
                list of selected keys of self.read_losses

            Returns
            -------
            ensemble: StackingEnsemble
                trained Ensemble
        """

        if self.unit_test:
            raise MemoryError()

        predictions_train = [self.read_preds[k][Y_ENSEMBLE] for k in ensemble_identifiers]
        best_model_predictions = self.read_preds[best_model_identifier][Y_ENSEMBLE]

        ensemble_num_runs = [
            (
                self.read_losses[k]["seed"],
                self.read_losses[k]["num_run"],
                self.read_losses[k]["budget"],
            )
            for k in ensemble_identifiers]

        best_model_num_run = (
            self.read_losses[best_model_identifier]["seed"],
            self.read_losses[best_model_identifier]["num_run"],
            self.read_losses[best_model_identifier]["budget"],
        )

        # check hash if ensemble training data changed
        current_hash = "".join([
            str(zlib.adler32(predictions_train[i].data.tobytes()))
            for i in range(len(predictions_train))
        ])
        if self.last_hash == current_hash:
            self.logger.debug(
                "No new model predictions selected -- skip ensemble building "
                "-- current performance: %f",
                self.validation_performance_,
            )

            return None
        self.last_hash = current_hash

        opt_metric = [m for m in self.metrics if m.name == self.opt_metric][0]
        if not opt_metric:
            raise ValueError(f"Cannot optimize for {self.opt_metric} in {self.metrics} "
                             "as more than one unique optimization metric was found.")

        ensemble = StackingEnsemble(
            ensemble_size=self.ensemble_size,
            metric=opt_metric,
            random_state=self.random_state,
            task_type=self.task_type,
        )

        try:
            self.logger.debug(
                "Fitting the ensemble on %d models.",
                len(predictions_train),
            )
            start_time = time.time()
            ensemble.fit(
                predictions_train, 
                best_model_predictions,
                self.y_true_ensemble,
                ensemble_num_runs,
                best_model_num_run)
            end_time = time.time()
            self.logger.debug(
                "Fitting the ensemble took %.2f seconds.",
                end_time - start_time,
            )
            self.logger.info(str(ensemble))
            self.validation_performance_ = min(
                self.validation_performance_,
                ensemble.get_validation_performance(),
            )

        except ValueError:
            self.logger.error('Caught ValueError: %s', traceback.format_exc())
            return None
        except IndexError:
            self.logger.error('Caught IndexError: %s' + traceback.format_exc())
            return None
        finally:
            # Explicitly free memory
            del predictions_train

        return ensemble

    def predict(self, set_: str,
                ensemble: AbstractEnsemble,
                selected_keys: list,
                n_preds: int,
                index_run: int) -> np.ndarray:
        """
            save preditions on ensemble, validation and test data on disc
            Parameters
            ----------
            set_: ["test"]
                data split name
            ensemble: EnsembleSelection
                trained Ensemble
            selected_keys: list
                list of selected keys of self.read_losses
            n_preds: int
                number of prediction models used for ensemble building
                same number of predictions on valid and test are necessary
            index_run: int
                n-th time that ensemble predictions are written to disc
            Return
            ------
            y: np.ndarray
        """
        self.logger.debug("Predicting the %s set with the ensemble!", set_)

        if set_ == 'test':
            pred_set = Y_TEST
        else:
            pred_set = Y_ENSEMBLE
        predictions = [self.read_preds[k][pred_set] for k in selected_keys]

        if n_preds == len(predictions):
            y = ensemble.predict(predictions)
            if self.output_type == BINARY:
                y = y[:, 1]
            if self.SAVE2DISC:
                self.backend.save_predictions_as_txt(
                    predictions=y,
                    subset=set_,
                    idx=index_run,
                    prefix=self.dataset_name,
                    precision=8,
                )
            return y
        else:
            self.logger.info(
                "Found inconsistent number of predictions and models (%d vs "
                "%d) for subset %s",
                len(predictions),
                n_preds,
                set_,
            )
            return None

    def get_n_best_preds(self) -> List[str]:
        """
            get best n predictions (i.e., keys of self.read_losses)
            according to the loss on the "ensemble set"
            n: all models

            Side effects:
                ->Define the n-best models to use in ensemble
                ->Only the best models are loaded
                ->Any model that is not best is candidate to deletion
                  if max models in disc is exceeded.
        """

        sorted_keys = self._get_list_of_sorted_preds()

        # number of models available
        num_keys = len(sorted_keys)
        # remove all that are at most as good as random
        # note: dummy model must have run_id=1 (there is no run_id=0)
        dummy_losses = list(filter(lambda x: x[2] == 1, sorted_keys))
        # Leave this here for when we enable dummy classifier/scorer
        if len(dummy_losses) > 0:
            # number of dummy models
            num_dummy = len(dummy_losses)
            dummy_loss = dummy_losses[0]
            self.logger.debug("Use %f as dummy loss" % dummy_loss[1])
            sorted_keys = list(filter(lambda x: x[1] < dummy_loss[1], sorted_keys))

            # remove Dummy Classifier
            sorted_keys = list(filter(lambda x: x[2] > 1, sorted_keys))
            if len(sorted_keys) == 0:
                # no model left; try to use dummy loss (num_run==0)
                # log warning when there are other models but not better than dummy model
                if num_keys > num_dummy:
                    self.logger.warning("No models better than random - using Dummy Score!"
                                        "Number of models besides current dummy model: %d. "
                                        "Number of dummy models: %d",
                                        num_keys - 1,
                                        num_dummy)
                sorted_keys = [
                    (k, v["ens_loss"], v["num_run"]) for k, v in self.read_losses.items()
                    if v["seed"] == self.seed and v["num_run"] == 1
                ]

        # reduce to keys
        reduced_sorted_keys = list(map(lambda x: x[0], sorted_keys))

        # Load the predictions for the winning
        for k in reduced_sorted_keys:
            if (
                (
                    k not in self.read_preds or self.read_preds[k][Y_ENSEMBLE] is None
                )
                and self.read_losses[k]['loaded'] != 3
            ):
                self.read_preds[k][Y_ENSEMBLE] = self._read_np_fn(k)
                # No need to load test here because they are loaded
                #  only if the model ends up in the ensemble
                self.read_losses[k]['loaded'] = 1
        # return best scored keys of self.read_losses
        return reduced_sorted_keys

    def get_ensemble_loss(self, ensemble_identifiers: List[str], model_predictions: np.ndarray):
        """
        Gets the loss of the ensemble given slot j and predictions for new model at slot j
        set is ensemble
        Args:
            ensemble_identifiers ([type]): [description]
            model_predictions ([type]): [description]
        """


        if ensemble_identifiers is None:
            loss = calculate_loss(
                    metrics=[self.metric],
                    target=self.y_true_ensemble,
                    prediction=model_predictions,
                    task_type=self.task_type,
                )
        else:
            weighted_ensemble_prediction = np.zeros(
                model_predictions.shape,
                dtype=np.float64,
            )
            fant_ensemble_prediction = np.zeros(
                weighted_ensemble_prediction.shape,
                dtype=np.float64,
            )

            
            for i, identifier in enumerate(ensemble_identifiers):
                if self.read_preds[identifier][Y_ENSEMBLE] == None:
                    # y ensemble read_preds is loaded in get_n_best_preds. If there is no value for this that means its a new model at this iteration.
                    raise ValueError("check here to resolve starting condition")
                predictions = self.read_preds[identifier][Y_ENSEMBLE] if i != self.ensemble_slot_j else model_predictions

                np.add(
                    weighted_ensemble_prediction,
                    predictions,
                    out=fant_ensemble_prediction
                )
                np.multiply(
                    fant_ensemble_prediction,
                    (1. / float(self.ensemble_size)),
                    out=fant_ensemble_prediction
                )
            loss = calculate_loss(
                        metrics=[self.metric],
                        target=self.y_true_ensemble,
                        prediction=fant_ensemble_prediction,
                        task_type=self.task_type,
                    )
        return loss