import os as os
import json
import numpy as np
import time
import random
import logging
import pickle
from abc import abstractmethod

from sklearn.model_selection import train_test_split


class BaseBaseline():

    def __init__(self, name):

        self.configure_logging()

        self.name = name
        self.config = self.get_config()

        self.categoricals = None
        self.all_nan = None
        self.encode_dicts = None
        self.num_classes = None

    def configure_logging(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        self.logger.addHandler(ch)

    def get_config(self):
        dirname = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(dirname, "baseline_configs", self.name + ".json")
        with open(config_path, "r") as f:
            config = json.load(f)
        for k,v in config.items():
            if v=="True":
                config[k] = True
            if v=="False":
                config[k] = False
        return config

    def save(self, model_path, info_path):
        info_dict = {"nan_cols": self.all_nan,
                     "encode_dict": self.encode_dicts,
                     "categoricals": self.categoricals,
                     "model_name": self.name,
                     "num_classes": self.num_classes}

        pickle.dump(info_dict, open(info_path, "wb"))
        pickle.dump(self.model, open(model_path, "wb"))

    def load(self, model_path, info_path):

        info = pickle.load(open(info_path, "rb"))

        #self.name = info["model_name"]
        self.all_nan = info["nan_cols"]
        self.categoricals = info["categoricals"]
        self.encode_dicts = info["encode_dict"]
        self.num_classes = info["num_classes"]

        self.model = pickle.load(open(model_path, "rb"))

    @abstractmethod
    def fit(self, X_train, y_train, X_val, y_val):
        pass

    @abstractmethod
    def score(self, X_test, y_test):
        pass
