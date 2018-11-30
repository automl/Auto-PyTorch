__author__ = "Max Dippel, Michael Burkart and Matthias Urban"
__version__ = "0.0.1"
__license__ = "BSD"

from autonet.pipeline.base.pipeline_node import PipelineNode
from autonet.utils.config.config_option import ConfigOption, to_bool
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np
import scipy.sparse

class OneHotEncoding(PipelineNode):
    def __init__(self):
        super(OneHotEncoding, self).__init__()
        self.encode_Y = False

    def fit(self, pipeline_config, X_train, X_valid, Y_train, Y_valid, categorical_features):
        ohe = OneHotEncoder(categories="auto", sparse=False, handle_unknown="ignore")
        encoder = ColumnTransformer(transformers=[("ohe", ohe, [i for i, f in enumerate(categorical_features) if f])], remainder="passthrough")
        encoder.categories_ = np.array([])
        encoder.categorical_features = categorical_features

        if any(categorical_features) and not scipy.sparse.issparse(X_train):
            # encode X
            X_train = encoder.fit_transform(X_train)
            if (X_valid is not None):
                X_valid = encoder.transform(X_valid)
            encoder.categories_ = encoder.transformers_[0][1].categories_

        # Y to matrix
        y_encoder = None
        Y_train = Y_train.astype(np.float32)
        if len(Y_train.shape) == 1:
            Y_train = Y_train.reshape(-1, 1)
        if Y_valid is not None and len(Y_valid.shape) == 1:
            Y_valid = Y_valid.reshape(-1, 1)

        # encode Y
        if self.encode_Y and not scipy.sparse.issparse(Y_train):
            y_encoder = OneHotEncoder(sparse=False, categories="auto", handle_unknown='ignore')
            y_encoder.categories_ = np.array([])
            Y_train = y_encoder.fit_transform(Y_train)
            if Y_valid is not None:
                Y_valid = y_encoder.transform(Y_valid)

        return {'X_train': X_train, 'X_valid': X_valid, 'one_hot_encoder': encoder, 'Y_train': Y_train, 'Y_valid': Y_valid,
            'y_one_hot_encoder': y_encoder, 'categorical_features': None}

    def predict(self, pipeline_config, X, one_hot_encoder):
        categorical_features = pipeline_config["categorical_features"]
        if categorical_features and any(categorical_features) and not scipy.sparse.issparse(X):
            X = one_hot_encoder.transform(X)
        return {'X': X, 'one_hot_encoder': one_hot_encoder}

    def get_pipeline_config_options(self):
        options = [
            ConfigOption(name='categorical_features', default=[], type=to_bool, list=True,
                info='List of booleans that specifies for each feature whether it is categorical.')
        ]
        return options
    
    def reverse_transform_y(self, Y, y_one_hot_encoder):
        if y_one_hot_encoder is None:
            return Y
        return y_one_hot_encoder.categories_[0][np.argmax(Y, axis=1)].reshape(-1, 1)
    
    def transform_y(self, Y, y_one_hot_encoder):
        if y_one_hot_encoder is None:
            return Y
        return y_one_hot_encoder.transform(Y.reshape(-1, 1))