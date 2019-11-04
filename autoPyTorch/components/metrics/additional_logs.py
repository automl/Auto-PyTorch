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


class gradient_norm():
    """Log the norm of the gradients"""
    def __init_(self):
        pass

    def __call__(self, network, epoch):
        total_gradient = 0
        n_params = 0

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
            total_gradient += p.grad.data.norm(2).item()
            n_params += 1

        # Prevent division through 0
        if total_gradient==0:
            n_params = 1

        return total_gradient/n_params
