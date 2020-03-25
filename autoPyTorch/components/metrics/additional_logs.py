import warnings
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score


def ensure_numpy(y):
    if type(y)==torch.Tensor:
        return y.detach().cpu().numpy()
    return y


class GradientLogger(object):

    def __init__(self):

        self.current_epoch = 0
        self.current_gradients = None

    def _update(self, network, epoch):
        if epoch > self.current_epoch or self.current_gradients is None:
            self.current_gradients = self._get_gradient_list(network)
            self.current_epoch = epoch

    def _get_gradient_list(self, network):
        """Returns all gradients stored in a nn.Module as np.array"""
        all_grads = []

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
                for par in torch.flatten(p.grad.data):
                    all_grads.append(par.cpu().numpy())

        if all_grads==[]:
            all_grads.append(0)

        return np.array(all_grads)

    def get_gradients(self, network, epoch):
        self._update(network, epoch)
        return self.current_gradients


class CustomGradientLogger():
    """
    Creates a logger which applies a given function to the gradients of an nn.Module. Gradient accumulation is
    shared across multiple instances if the same GradientLogger instance is passed to the init.
    """

    def __init__(self, gradient_logger_instance, log_func):

        warnings.warn("Currently CustomGradientLogger is not mean to be used for different models (search)", DeprecationWarning, stacklevel=2)

        self.gradient_logger = gradient_logger_instance
        self.log_func = log_func

    def __call__(self, network, epoch):
        return self.log_func.__call__(self.gradient_logger.get_gradients(network, epoch))


class gradient_max(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_max, self).__init__(gradient_logger_instance, np.max)


class gradient_mean(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_mean, self).__init__(gradient_logger_instance, np.mean)


class gradient_median(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_median, self).__init__(gradient_logger_instance, np.median)


class gradient_std(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_std, self).__init__(gradient_logger_instance, np.std)


class gradient_q10(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_q10, self).__init__(gradient_logger_instance, lambda x: np.percentile(x,10))


class gradient_q25(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_q25, self).__init__(gradient_logger_instance, lambda x: np.percentile(x,25))


class gradient_q75(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_q75, self).__init__(gradient_logger_instance, lambda x: np.percentile(x,75))


class gradient_q90(CustomGradientLogger):

    def __init__(self, gradient_logger_instance):
        super(gradient_q90, self).__init__(gradient_logger_instance, lambda x: np.percentile(x,90))


class LayerWiseGradientLogger(object):

    def __init__(self):

        self.current_epoch = 0
        self.current_layer_wise_gradients = None

    def _update(self, network, epoch):
        if epoch > self.current_epoch or self.current_layer_wise_gradients is None:
            self.current_layer_wise_gradients = self._get_layer_wise_gradients(network)
            self.current_epoch = epoch

    def _get_layer_wise_gradients(self, network):
        layer_wise_grads = []

        for layer in network.modules():
            if isinstance(layer, (nn.Linear, nn.Conv2d)):
                layer_grads = self._get_gradient_list(layer)
                layer_wise_grads.append(layer_grads)

        return layer_wise_grads

    def _get_gradient_list(self, network):
        """Returns all gradients stored in a nn.Module as np.array"""
        all_grads = []

        for p in list(filter(lambda p: p.grad is not None, network.parameters())):
                for par in torch.flatten(p.grad.data):
                    all_grads.append(par.cpu().numpy())

        if all_grads==[]:
            all_grads.append(0)

        return all_grads

    def get_layer_wise_gradients(self, network, epoch):
        self._update(network, epoch)
        return self.current_layer_wise_gradients


class CustomLayerWiseGradientLogger():
    """
    Creates a logger which applies a given function to the gradients of an nn.Module. Gradients are stored as a
    list of lists. Gradient accumulation is shared across multiple instances if the same LayerWiseGradientLogger
    instance is passed to the init.
    """

    def __init__(self, layer_wise_gradient_logger_instance, log_func):

        warnings.warn("Currently CustomLayerWiseGradientLogger is not mean to be used for different models (search)", DeprecationWarning, stacklevel=2)

        self.layer_wise_gradient_logger_instance = layer_wise_gradient_logger_instance
        self.log_func = log_func

    def __call__(self, network, epoch):
        output = []
        for layer_grads in self.layer_wise_gradient_logger_instance.get_layer_wise_gradients(network, epoch):
            output.append(self.log_func.__call__(np.array(layer_grads)))
        return np.array(output)


class layer_wise_gradient_max(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_max, self).__init__(layer_wise_gradient_logger_instance, np.max)


class layer_wise_gradient_mean(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_mean, self).__init__(layer_wise_gradient_logger_instance, np.mean)


class layer_wise_gradient_median(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_median, self).__init__(layer_wise_gradient_logger_instance, np.median)


class layer_wise_gradient_std(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_std, self).__init__(layer_wise_gradient_logger_instance, np.std)


class layer_wise_gradient_q10(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_q10, self).__init__(layer_wise_gradient_logger_instance, lambda x: np.percentile(x, 10))


class layer_wise_gradient_q25(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_q25, self).__init__(layer_wise_gradient_logger_instance, lambda x: np.percentile(x, 25))


class layer_wise_gradient_q75(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_q75, self).__init__(layer_wise_gradient_logger_instance, lambda x: np.percentile(x, 75))


class layer_wise_gradient_q90(CustomLayerWiseGradientLogger):

    def __init__(self, layer_wise_gradient_logger_instance):
        super(layer_wise_gradient_q90, self).__init__(layer_wise_gradient_logger_instance, lambda x: np.percentile(x, 90))


class test_result():
    """Log the performance on the test set"""
    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test
    
    def __call__(self, network, epochs):
        if self.Y_test is None and self.X_test is None:
            return float("nan")
        
        return self.autonet.score(self.X_test, self.Y_test)


class test_cross_entropy():

    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test

    def __call__(self, network, epoch):
        if self.Y_test is None or self.X_test is None:
            return float("nan")

        _, test_predictions = self.autonet.predict(self.X_test, return_probabilities=True)
        loss_func = nn.CrossEntropyLoss()
        return loss_func(torch.from_numpy(test_predictions), torch.from_numpy(self.Y_test))


class test_balanced_accuracy():

    def __init__(self, autonet, X_test, Y_test):
        self.autonet = autonet
        self.X_test = X_test
        self.Y_test = Y_test

    def __call__(self, network, epoch):
        if self.Y_test is None or self.X_test is None:
            return float("nan")

        test_predictions = self.autonet.predict(self.X_test)

        return balanced_accuracy_score(torch.from_numpy(self.Y_test), torch.from_numpy(test_predictions))


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
