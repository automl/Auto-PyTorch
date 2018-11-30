__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"


import os
import sys
import logging

from autoPyTorch import AutoNetClassification

from autoPyTorch.data_management.data_manager import DataManager

from autoPyTorch.utils.mem_test_thread import MemoryLogger

dm = DataManager(verbose=1)
dataset_dir = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'datasets'))


dm.read_data(os.path.join(dataset_dir, "classification/dataset_28_optdigits.csv"), is_classification=True)
# 5620 samples, 10 classes, 65 features      --->    98% validation accuracy







mem_logger = MemoryLogger()
mem_logger.start()

try:
    autonet = AutoNetClassification(early_stopping_patience=15, budget_type='epochs', min_budget=1, max_budget=9, num_iterations=1, log_level='error')

    res = autonet.fit(X_train=dm.X,
                        Y_train=dm.Y,
                        X_valid=dm.X_train,
                        Y_valid=dm.Y_train,
                        categorical_features=dm.categorical_features)
    print(res)
    
finally:
    mem_logger.stop()

