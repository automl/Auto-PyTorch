import ConfigSpace

class PreprocessorBase():
    def __init__(self, hyperparameter_config):
        pass


    def fit(self, X, Y):
        """Fit preprocessor with X and Y.
        
        Arguments:
            X {tensor} -- feature matrix
            Y {tensor} -- labels
        """
        pass

    def transform(self, X, **kwargs):
        """Preprocess X
        
        Arguments:
            X {tensor} -- feature matrix
        
        Returns:
            X -- preprocessed X
        """

        return X

    @staticmethod
    def get_hyperparameter_search_space():
        return ConfigSpace.ConfigurationSpace()


    