import numpy as np
import torch

from sklearn.metrics import accuracy_score

class test_result_ens():
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
        from autoPyTorch.core.api import AutoNet
        self.predict = AutoNet.predict

    def __call__(self, model, epochs):
        if self.Y_test is None or self.X_test is None:
            return float("nan")

        preds = self.predict(self.autonet, self.X_test, return_probabilities=False)
        return accuracy_score(preds, self.Y_test)

class test_result():
    """Log the performance on the test set"""
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
    
    def __call__(self, network, epochs):
        if self.Y_test is None or self.X_test is None:
            return float("nan")
        
        return self.autonet.score(self.X_test, self.Y_test)


class gradient_norm():
    """Log the mean norm of the loss gradients"""
    def __init_(self):
        pass

    def __call__(self, network, epoch):
        total_gradient = 0
        n_params = 0

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
            total_gradient += p.grad.data.norm(2).item()
            n_params += 1

        # Prevent dividing by 0
        if total_gradient==0:
            n_params = 1

        return total_gradient/n_params


class gradient_mean():
    """Log the mean of the loss gradients"""
    def __init_(self):
        pass

    def __call__(self, network, epoch):

        n_gradients = 0
        sum_of_means = 0

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
            weight = np.prod(p.grad.data.shape)
            n_gradients += weight
            sum_of_means += p.grad.data.mean().item() * weight

        # Prevent dividing by 0
        if n_gradients==0:
            n_gradients = 1

        return sum_of_means/n_gradients


class gradient_std():
    """Log the norm of the loss gradients"""

    def __init_(self):
        pass

    def __call__(self, network, epoch):

        n = 0
        total_sum = 0
        sum_sq = 0

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
            for par in torch.flatten(p.grad.data):
                n += 1
                total_sum += par
                sum_sq += par*par

        # Prevent dividing by 0
        if n==0:
            n = 1

        return (sum_sq - (total_sum * total_sum) / n) / (n-1)
