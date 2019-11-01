from __future__ import print_function, division
import pandas as pd
import numpy as np
from abc import abstractmethod
import os
from  scipy.sparse import csr_matrix
import math

from autoPyTorch.data_management.data_converter import DataConverter

__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

class DataReader(object):
    def __init__(self, file_name, is_classification=None):
        self.file_name = file_name
        self.data = None
        self.X = None
        self.Y = None
        self.X_valid = None
        self.Y_valid = None
        self.X_test = None
        self.Y_test = None
        self.is_classification = is_classification
        self.categorical_features = None
        self.is_multilabel = None
        self.max_runtime = None
        self.metric = None

    @abstractmethod
    def read(self):
        return

    def convert(self, **kwargs):
        """
        Convert the data using standard data converter with standard settings.
        
        Arguments:
            **kwargs: args and kwargs are passed to Dataconverter
        """
        data_converter = DataConverter(is_classification=self.is_classification, is_multilabel=self.is_multilabel, **kwargs)
        self.X, self.Y, self.is_classification, self.is_multilabel, self.categorical_features = data_converter.convert(self.X, self.Y)

        if self.X_valid is not None and self.Y_valid is not None:
            self.X_valid, self.Y_valid, _, _, _ = data_converter.convert(self.X_valid, self.Y_valid)

        if self.X_test is not None and self.Y_test is not None:
            self.X_test, self.Y_test, _, _, _ = data_converter.convert(self.X_test, self.Y_test)


class CSVReader(DataReader):
    def __init__(self, file_name, is_classification=None):
        self.num_entries = None
        self.num_features = None
        self.num_classes = None
        super(CSVReader, self).__init__(file_name, is_classification)
        
    

    def read(self, auto_convert=True, **kwargs):
        """
        Read the data from given csv file.
        
        Arguments:
            auto_convert: Automatically convert data after reading.
            *args, **kwargs: arguments for converting.
        """
        self.data = pd.read_csv(self.file_name)


        self.num_entries = len(self.data)
        self.num_features = len(self.data.iloc[0]) - 1

        self.data = np.array(self.data)
            
        self.X = self.data[0:self.num_entries, 0:self.num_features] #np.array(  .iloc
        self.Y = self.data[0:self.num_entries, -1]

        for i in range(self.X.shape[0]):
            for j in range(self.X.shape[1]):
                if self.X[i, j] == "?":
                    self.X[i, j] = np.nan

        self.num_classes = len(np.unique(self.Y))
        if (auto_convert):
            self.convert(**kwargs)
            
class OpenMlReader(DataReader):
    def __init__(self, dataset_id, is_classification = None, api_key=None):
        import openml
        self.openml = openml
        self.num_entries = None
        self.num_features = None
        self.num_classes = None
        self.dataset_id = dataset_id
        if api_key:
            openml.config.server = "https://www.openml.org/api/v1/xml"
            openml.config.apikey = api_key
        super(OpenMlReader, self).__init__("openml:" + str(dataset_id), is_classification)

    def read(self, **kwargs):
        """
        Read the data from given openml dataset file.
        
        Arguments:
            auto_convert: Automatically convert data after reading.
            *args, **kwargs: arguments for converting.
        """
        
        dataset = self.openml.datasets.get_dataset(self.dataset_id)
        try:
            self.X, self.Y, self.categorical_features = dataset.get_data(
                target=dataset.default_target_attribute, return_categorical_indicator=True)
        except Exception as e:
            raise RuntimeError("An error occurred when loading the dataset and splitting it into X and Y. Please check if the dataset is suitable.")

        self.num_entries = self.X.shape[0]
        self.num_features = self.X.shape[1]
        self.is_multilabel = False
        class_labels = dataset.retrieve_class_labels(target_name=dataset.default_target_attribute)
        if class_labels:
            self.is_classification = True
            self.num_classes = len(class_labels)
        else:
            self.is_classification = False
            self.num_classes = 1


class AutoMlReader(DataReader):
    def __init__(self, path_to_info):
        self.num_entries = None
        self.num_features = None
        self.num_classes = None
        super(AutoMlReader, self).__init__(path_to_info, None)
    
    def read(self, auto_convert=True, **kwargs):
        path_to_info = self.file_name
        info_dict = dict()

        # read info file
        with open(path_to_info, "r") as f:
            for line in f:
                info_dict[line.split("=")[0].strip()] = line.split("=")[1].strip().strip("'")
        self.is_classification = "classification" in info_dict["task"]
        
        name = info_dict["name"]
        path = os.path.dirname(path_to_info)
        self.is_multilabel = "multilabel" in info_dict["task"] if self.is_classification else None
        self.metric = info_dict["metric"]
        self.max_runtime = float(info_dict["time_budget"])

        target_num = int(info_dict["target_num"])
        feat_num = int(info_dict["feat_num"])
        train_num = int(info_dict["train_num"])
        valid_num = int(info_dict["valid_num"])
        test_num = int(info_dict["test_num"])
        is_sparse = bool(int(info_dict["is_sparse"]))
        feats_binary = info_dict["feat_type"].lower() == "binary"

        # read feature types
        force_categorical = []
        force_numerical = []
        if info_dict["feat_type"].lower() == "binary" or info_dict["feat_type"].lower() == "numerical":
            force_numerical = [i for i in range(feat_num)]
        elif info_dict["feat_type"].lower() == "categorical":
            force_categorical = [i for i in range(feat_num)]
        elif os.path.exists(os.path.join(path, name + "_feat.type")):
            with open(os.path.join(path, name + "_feat.type"), "r") as f:
                for i, line in enumerate(f):
                    if line.strip().lower() == "numerical":
                        force_numerical.append(i)
                    elif line.strip().lower() == "categorical":
                        force_categorical.append(i)
        
        # read data files
        reading_function = self.read_datafile if not is_sparse else (
            self.read_sparse_datafile if not feats_binary else self.read_binary_sparse_datafile)
        self.X = reading_function(os.path.join(path, name + "_train.data"), (train_num, feat_num))
        self.Y = self.read_datafile(os.path.join(path, name + "_train.solution"), (train_num, target_num))

        if os.path.exists(os.path.join(path, name + "_valid.data")) and \
            os.path.exists(os.path.join(path, name + "_valid.solution")):
            self.X_valid = reading_function(os.path.join(path, name + "_valid.data"), (valid_num, feat_num))
            self.Y_valid = self.read_datafile(os.path.join(path, name + "_valid.solution"), (valid_num, target_num))
        
        if os.path.exists(os.path.join(path, name + "_test.data")) and \
            os.path.exists(os.path.join(path, name + "_test.solution")):
            self.X_test = reading_function(os.path.join(path, name + "_test.data"), (test_num, feat_num))
            self.Y_test = self.read_datafile(os.path.join(path, name + "_test.solution"), (test_num, target_num))
        
        if not self.is_multilabel and self.is_classification and self.Y.shape[1] > 1:
            self.Y = np.argmax(self.Y, axis=1)
            self.Y_valid = np.argmax(self.Y_valid, axis=1) if self.Y_valid is not None else None
            self.Y_test = np.argmax(self.Y_test, axis=1) if self.Y_test is not None else None

        if auto_convert and not is_sparse:
            self.convert(force_categorical=force_categorical, force_numerical=force_numerical, **kwargs)
        
    def read_datafile(self, filepath, shape):
        data = []
        with open(filepath, "r") as f:
            for line in f:
                data.append([float(v.strip()) for v in line.split()])
        return np.array(data)

    def read_sparse_datafile(self, filepath, shape):
        data = []
        row_indizes = []
        col_indizes = []
        with open(filepath, "r") as f:
            for row, line in enumerate(f):
                print("\rReading line:",  row, "of", shape[0], end="")
                for value in line.split():
                    value = value.rstrip()

                    data.append(float(value.split(":")[1]))
                    col_indizes.append(int(value.split(":")[0]) - 1)
                    row_indizes.append(row)
            print("Done")
        return csr_matrix((data, (row_indizes, col_indizes)), shape=shape)
    
    def read_binary_sparse_datafile(self, filepath, shape):
        row_indizes = []
        col_indizes = []
        with open(filepath, "r") as f:
            for row, line in enumerate(f):
                print("\rReading line:",  row, "of", shape[0], end="")
                for value in line.split():
                    value = value.rstrip()
                    col_indizes.append(int(value) - 1)
                    row_indizes.append(row)
            print("Done")
        return csr_matrix(([1] * len(row_indizes), (row_indizes, col_indizes)), shape=shape)


class OpenMLImageReader(OpenMlReader):
    def __init__(self, dataset_id, is_classification = None, api_key=None, nChannels=1):
        self.channels = nChannels
        super(OpenMLImageReader, self).__init__(dataset_id, is_classification, api_key)

    def read(self, auto_convert=True, **kwargs):
        """
        Read the data from given openml datset file.
        
        Arguments:
            auto_convert: Automatically convert data after reading.
            *args, **kwargs: arguments for converting.
        """
        
        dataset = self.openml.datasets.get_dataset(self.dataset_id)
        self.data = dataset.get_data()


        self.num_entries = len(self.data)
        self.num_features = len(self.data[0]) - 1

            
        self.X = self.data[0:self.num_entries, 0:self.num_features] / 255

        image_size = int(math.sqrt(self.num_features / self.channels))
        self.X = np.reshape(self.X, (self.X.shape[0], self.channels, image_size, image_size))
        
        self.Y = self.data[0:self.num_entries, -1]
        self.num_classes = len(np.unique(self.Y))
        if self.is_classification is None:
            self.is_classification = dataset.get_features_by_type("nominal")[-1] == self.num_features
