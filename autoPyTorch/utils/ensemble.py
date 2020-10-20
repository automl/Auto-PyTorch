import os
import time
import numpy as np
import json
import math
import tempfile
import uuid
import asyncio
import multiprocessing
import signal
import logging
from autoPyTorch.components.ensembles.ensemble_selection import EnsembleSelection

def build_ensemble(result, optimize_metric,
        ensemble_size, all_predictions, labels, model_identifiers,
        only_consider_n_best=0, sorted_initialization_n_best=0):
    id2config = result.get_id2config_mapping()
    ensemble_selection = EnsembleSelection(ensemble_size, optimize_metric,
        only_consider_n_best=only_consider_n_best, sorted_initialization_n_best=sorted_initialization_n_best)

    # fit ensemble
    ensemble_selection.fit(np.array(all_predictions), labels, model_identifiers)
    ensemble_configs = dict()
    for identifier in ensemble_selection.get_selected_model_identifiers():
        try:
            ensemble_configs[tuple(identifier[:3])] = id2config[tuple(identifier[:3])]["config"]
        except:
            #TODO: Do this properly (baseline configs are not logged by bohb)
            ensemble_configs[tuple(identifier[:3])] = {"model": "baseline"}
    return ensemble_selection, ensemble_configs


def read_ensemble_prediction_file(filename, y_transform):
    all_predictions = list()
    all_timestamps = list()
    labels = None
    model_identifiers = list()
    with open(filename, "rb") as f:
        labels = np.load(f, allow_pickle=True)
        labels, _ = y_transform(labels)

        while True:
            try:
                job_id, budget, timestamps = np.load(f, allow_pickle=True)
                predictions = np.load(f, allow_pickle=True)
                model_identifiers.append(job_id + (budget, ))
                predictions = np.array(predictions)
                all_predictions.append(predictions)
                all_timestamps.append(timestamps)
            except (EOFError, OSError):
                break
    return all_predictions, labels, model_identifiers, all_timestamps


class test_predictions_for_ensemble():
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
        from autoPyTorch.core.api import AutoNet
        self.predict = AutoNet.predict

    
    def __call__(self, model, epochs):
        if self.Y_test is None or self.X_test is None:
            return float("nan")
        
        return self.predict(self.autonet, self.X_test, return_probabilities=True)[1], self.Y_test

def combine_predictions(data, pipeline_kwargs, X, Y):
    all_indices = None
    all_predictions = None
    for split, predictions in data.items():
        if (np.any(np.isnan(predictions))):
            logging.getLogger("autonet").warn("Not saving predictions containing nans")
            return None
        indices = pipeline_kwargs[split]["valid_indices"]
        assert len(predictions) == len(indices), "Different number of predictions and indices:" + str(len(predictions)) + "!=" + str(len(indices))
        all_indices = indices if all_indices is None else np.append(all_indices, indices)
        all_predictions = predictions if all_predictions is None else np.vstack((all_predictions, predictions))
    argsort = np.argsort(all_indices)
    sorted_predictions = all_predictions[argsort]
    sorted_indices = all_indices[argsort]
    
    unique = uuid.uuid4()
    tempfile.gettempdir()
    with open(os.path.join(tempfile.gettempdir(), "autonet_ensemble_predictions_%s.npy" % unique), "wb") as f:
        np.save(f, sorted_predictions, allow_pickle=True)
    with open(os.path.join(tempfile.gettempdir(), "autonet_ensemble_labels_%s.npy" % unique), "wb") as f:
        np.save(f, Y[sorted_indices], allow_pickle=True)
    host, port = pipeline_kwargs[0]["pipeline_config"]["ensemble_server_credentials"]
    return host, port, unique

def combine_test_predictions(data, pipeline_kwargs, X, Y):
    predictions = [d[0] for d in data.values() if d == d]
    labels = [d[1] for d in data.values() if d == d]
    assert all(np.all(labels[0] == l) for l in labels[1:])
    assert len(predictions) == len(labels)
    if len(predictions) == 0:
        return None
    
    unique = uuid.uuid4()
    tempfile.gettempdir()
    with open(os.path.join(tempfile.gettempdir(), "autonet_ensemble_predictions_%s.npy" % unique), "wb") as f:
        np.save(f, np.stack(predictions), allow_pickle=True)
    with open(os.path.join(tempfile.gettempdir(), "autonet_ensemble_labels_%s.npy" % unique), "wb") as f:
        np.save(f, labels[0], allow_pickle=True)
    host, port = pipeline_kwargs[0]["pipeline_config"]["ensemble_server_credentials"]
    return host, port, unique

def filter_nan_predictions(predictions, *args):
    nan_predictions = set([i for i, p in enumerate(predictions) if np.any(np.isnan(p))])
    return [
        [x for i, x in enumerate(vector) if i not in nan_predictions] if vector is not None else None
        for vector in [predictions, *args]
    ]

async def serve_predictions(reader, writer):
    data = await reader.read(1024)
    name, unique = data.decode().split("_")
    # logging.getLogger("autonet").info("Serve %s %s" % (name, unique))

    with open(os.path.join(tempfile.gettempdir(), "autonet_ensemble_%s_%s.npy" % (name, unique)), "rb") as f:
        while True:
            buf = f.read(1024)
            if not buf:
                break
            writer.write(buf)
    os.remove(os.path.join(tempfile.gettempdir(), "autonet_ensemble_%s_%s.npy" % (name, unique)))
    if name == "predictions" and os.path.exists(os.path.join(tempfile.gettempdir(), "autonet_ensemble_labels_%s.npy" % unique)):
        os.remove(os.path.join(tempfile.gettempdir(), "autonet_ensemble_labels_%s.npy" % unique))
    await writer.drain()
    writer.close()

def _start_server(host, queue):
    def shutdown(signum, stack):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, shutdown)
    loop = asyncio.get_event_loop()
    coro = asyncio.start_server(serve_predictions, host, 0, loop=loop)
    server = loop.run_until_complete(coro)
    host, port = server.sockets[0].getsockname()
    queue.put((host, port))
    try:
        loop.run_forever()
    except KeyboardInterrupt:
        pass
    server.close()
    loop.run_until_complete(server.wait_closed())
    loop.close()
    # logging.getLogger("autonet").info("Ensemble Server has been shut down")

def start_server(host):
    queue = multiprocessing.Queue()
    p = multiprocessing.Process(target=_start_server, args=(host, queue))
    p.start()
    host, port = queue.get()
    p.shutdown = p.terminate
    return host, port, p

class ensemble_logger(object):
    def __init__(self, directory, overwrite):
        self.start_time = time.time()
        self.directory = directory
        self.overwrite = overwrite
        self.labels_written = False
        self.test_labels_written = False
        
        self.file_name = os.path.join(directory, 'predictions_for_ensemble.npy')
        self.test_file_name = os.path.join(directory, 'test_predictions_for_ensemble.npy')

        try:
            with open(self.file_name, 'x') as fh: pass
        except FileExistsError:
            if overwrite:
                with open(self.file_name, 'w') as fh: pass
            else:
                raise FileExistsError('The file %s already exists.'%self.file_name)
        except:
            raise
        
        try:
            with open(self.test_file_name, 'x') as fh: pass
        except FileExistsError:
            if overwrite:
                with open(self.test_file_name, 'w') as fh: pass
            else:
                raise FileExistsError('The file %s already exists.'%self.test_file_name)
        except:
            raise

    def new_config(self, *args, **kwargs):
        pass
    
    async def save_remote_data(self, host, port, name, unique, f):
        remote_reader, remote_writer = await asyncio.open_connection(host, port)
        remote_writer.write(("%s_%s" % (name, unique)).encode())
        while not remote_reader.at_eof():
            f.write(await remote_reader.read(1024))
        remote_writer.close()

    def __call__(self, job):
        if job.result is None:
            return
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        if "predictions_for_ensemble" in job.result and job.result["predictions_for_ensemble"] is None and \
            "test_predictions_for_ensemble" in job.result and job.result["test_predictions_for_ensemble"] is not None:
            host, port, unique =  job.result["test_predictions_for_ensemble"]
            with open("/dev/null", "wb") as f:
                loop.run_until_complete(self.save_remote_data(host, port, "predictions", unique, f))

        #logging.info(job.result.__repr__()) #TODO: delete

        if "predictions_for_ensemble" in job.result and job.result["predictions_for_ensemble"] is not None:
            host, port, unique = job.result["predictions_for_ensemble"]
            #logging.info("==> Saving preds...") # #TODO: delete
            #if not self.labels_written:
            #    logging.info("==> (Labels)") #TODO: delete
            with open(self.file_name, "ab") as f:
                if not self.labels_written:
                    loop.run_until_complete(self.save_remote_data(host, port, "labels", unique, f))
                    self.labels_written = True
                np.save(f, np.array([job.id, job.kwargs['budget'], job.timestamps], dtype=object), allow_pickle=True)
                loop.run_until_complete(self.save_remote_data(host, port, "predictions", unique, f))
            del job.result["predictions_for_ensemble"]

            if "baseline_predictions_for_ensemble" in job.result and job.result["baseline_predictions_for_ensemble"] is not None:
                baseline_id = (int(job.result["info"]["baseline_id"]), 0, 0)
                host, port, unique = job.result["baseline_predictions_for_ensemble"]
                #logging.info("==> Saving baseline preds...") # #TODO: delete
                with open(self.file_name, "ab") as f:
                    if not self.labels_written:
                        raise RuntimeError("Baseline predictions found but no labels logged yet.")
                    np.save(f, np.array([baseline_id, 0., job.timestamps], dtype=object), allow_pickle=True)
                    loop.run_until_complete(self.save_remote_data(host, port, "predictions", unique, f))
                del job.result["baseline_predictions_for_ensemble"]

            if "test_predictions_for_ensemble" in job.result and job.result["test_predictions_for_ensemble"] is not None:
                host, port, unique =  job.result["test_predictions_for_ensemble"]
                with open(self.test_file_name, "ab") as f:
                    if not self.test_labels_written:
                         loop.run_until_complete(self.save_remote_data(host, port, "labels", unique, f))
                         self.test_labels_written = True
                    np.save(f, np.array([job.id, job.kwargs['budget'], job.timestamps], dtype=object), allow_pickle=True)
                    loop.run_until_complete(self.save_remote_data(host, port, "predictions", unique, f))
                del job.result["test_predictions_for_ensemble"]

            if "baseline_test_predictions_for_ensemble" in job.result and job.result["baseline_test_predictions_for_ensemble"] is not None:
                host, port, unique =  job.result["baseline_test_predictions_for_ensemble"]
                logging.info("==> Logging baseline test preds")
                with open(self.test_file_name, "ab") as f:
                    if not self.test_labels_written:
                         raise RuntimeError("Baseline test predictions found but no labels logged yet.")
                    np.save(f, np.array([baseline_id, 0., job.timestamps], dtype=object), allow_pickle=True)
                    loop.run_until_complete(self.save_remote_data(host, port, "predictions", unique, f))
                del job.result["baseline_test_predictions_for_ensemble"]
        loop.close()
