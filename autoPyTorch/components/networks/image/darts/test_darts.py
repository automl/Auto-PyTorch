from autoPyTorch.components.networks.image.darts.model import DARTSImageNet
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss


darts_config = {
        "ImageAugmentation:augment": "True", 
        "ImageAugmentation:cutout": "True", 
        "NetworkSelectorDatasetInfo:darts:auxiliary": True, 
        "NetworkSelectorDatasetInfo:darts:drop_path_prob": 0.2, 
        "NetworkSelectorDatasetInfo:darts:init_channels": 50, 
        "NetworkSelectorDatasetInfo:darts:layers": 20, 
        "NetworkSelectorDatasetInfo:network": "darts", 
        "CreateImageDataLoader:batch_size": 96, 
        "ImageAugmentation:autoaugment": "True", 
        "ImageAugmentation:cutout_holes": 1, 
        "ImageAugmentation:cutout_length": 16, 
        "ImageAugmentation:fastautoaugment": "False", 
        "LossModuleSelectorIndices:loss_module": "cross_entropy_weighted", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_0": "max_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_1": "sep_conv_5x5", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_0": "dil_conv_5x5", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_1": "sep_conv_3x3", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_normal_3": "0_1", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_normal_4": "0_1", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_normal_5": "2_3", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_reduce_3": "0_1", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_reduce_4": "0_2", 
        "NetworkSelectorDatasetInfo:darts:inputs_node_reduce_5": "1_3", 
        "OptimizerSelector:optimizer": "sgd", 
        "OptimizerSelector:sgd:learning_rate": 0.025, 
        "OptimizerSelector:sgd:momentum": 0.9, 
        "OptimizerSelector:sgd:weight_decay": 0.0003, 
        "SimpleLearningrateSchedulerSelector:cosine_annealing:T_max": 1500, 
        "SimpleLearningrateSchedulerSelector:cosine_annealing:eta_min": 1e-08, 
        "SimpleLearningrateSchedulerSelector:lr_scheduler": "cosine_annealing", 
        "SimpleTrainNode:batch_loss_computation_technique": "standard", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_11": "dil_conv_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_12": "skip_connect", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_2": "avg_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_3": "avg_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_5": "avg_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_normal_6": "avg_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_10": "max_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_12": "dil_conv_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_2": "max_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_3": "max_pool_3x3", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_5": "dil_conv_5x5", 
        "NetworkSelectorDatasetInfo:darts:edge_reduce_7": "avg_pool_3x3"}

config = dict()

for key,val in darts_config.items():
    if key.startswith("NetworkSelectorDatasetInfo:darts"):
        config[key.split(":")[-1]] = val


model = DARTSImageNet(config, in_features=(3,32,32), out_features=10, final_activation=None)
model.to("cuda:0")

test_data = torch.zeros((10,3,32,32)).to("cuda:0")
test_data_out = torch.empty(10, dtype=torch.long).random_(10).to("cuda:0")

criterion = CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
optimizer.zero_grad()
preds, preds_aux = model.forward(test_data)

loss = criterion(preds, test_data_out) + 0.4*criterion(preds_aux, test_data_out)
loss.backward()
optimizer.step()

print("Done")

"""
# Parameter testing
n_params = sum(p.numel() for p in model.parameters())
print(n_params, "baseline")

test = torch.zeros((1,3,64,64)).to("cuda:0")

model = DARTSImageNet(config, in_features=(3,32,32), out_features=10, final_activation=None)
model.to("cuda:0")
preds = model.forward(test)
n_params = sum(p.numel() for p in model.parameters())
print(n_params, "half resolution")

config["init_channels"] = 25
model = DARTSImageNet(config, in_features=(3,64,64), out_features=10, final_activation=None)
n_params = sum(p.numel() for p in model.parameters())
print(n_params, "half channels")

config["init_channels"] = 50
config["layers"] = 10
model = DARTSImageNet(config, in_features=(3,64,64), out_features=10, final_activation=None)
n_params = sum(p.numel() for p in model.parameters())
print(n_params, "half layers")
"""
