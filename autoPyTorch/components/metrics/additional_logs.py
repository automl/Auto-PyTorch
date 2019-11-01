
class test_result():
    """Log the performance on the test set"""
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
    
    def __call__(self, model, epochs):
        if self.Y_test is None or self.X_test is None:
            return float("nan")
        
        return self.autonet.score(self.X_test, self.Y_test)