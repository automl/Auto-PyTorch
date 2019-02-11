__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autoPyTorch.pipeline.base.pipeline_node import PipelineNode
from autoPyTorch.utils.config.config_option import ConfigOption, to_bool
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import scipy.sparse

class OneHotEncoding(PipelineNode):
    def __init__(self):
        super(OneHotEncoding, self).__init__()
        self.encode_Y = False

    def fit(self, pipeline_config, X, Y, dataset_info):
        categorical_features = dataset_info.categorical_features
        ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
        encoder = ColumnTransformer(transformers=[("ohe", ohe, [i for i, f in enumerate(categorical_features) if f])], remainder="passthrough")
        encoder.categories_ = np.array([])
        encoder.categorical_features = categorical_features

        if any(categorical_features) and not dataset_info.is_sparse:
            # encode X
            X = encoder.fit_transform(X)
            encoder.categories_ = encoder.transformers_[0][1].categories_

        # Y to matrix
        Y, y_encoder = self.complete_y_tranformation(Y)

        dataset_info.categorical_features = None
        return {'X': X, 'one_hot_encoder': encoder, 'Y': Y, 'y_one_hot_encoder': y_encoder, 'dataset_info': dataset_info}

    def predict(self, pipeline_config, X, one_hot_encoder):
        categorical_features = pipeline_config["categorical_features"]
        if categorical_features and any(categorical_features) and not scipy.sparse.issparse(X):
            X = one_hot_encoder.transform(X)
        return {'X': X, 'one_hot_encoder': one_hot_encoder}
    
    def reverse_transform_y(self, Y, y_one_hot_encoder):
        if y_one_hot_encoder is None:
            return Y
        return y_one_hot_encoder.categories_[0][np.argmax(Y, axis=1)].reshape(-1, 1)
    
    def transform_y(self, Y, y_one_hot_encoder):
        if y_one_hot_encoder is None:
            return Y
        return y_one_hot_encoder.transform(Y.reshape(-1, 1))
    
    def complete_y_tranformation(self, Y):
        # Y to matrix
        y_encoder = None
        Y = Y.astype(np.float32)
        if len(Y.shape) == 1:
            Y = Y.reshape(-1, 1)

        # encode Y
        if self.encode_Y:
            y_encoder = OneHotEncoder(sparse=False, categories="auto", handle_unknown='ignore')
            y_encoder.categories_ = np.array([])
            Y = y_encoder.fit_transform(Y)
        return Y, y_encoder