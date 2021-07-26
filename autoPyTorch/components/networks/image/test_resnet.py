import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
from resnet import ResNet


class stupid_dict(dict):

    def __init__(self, normal_dict):
        for key, val in normal_dict.items():
            self.__setitem__(key, val)

test_config = {"death_rate": 0.038439,
        "initial_filters": 30,
    "nr_main_blocks": 3,
    "nr_residual_blocks_1": 3,
    "nr_residual_blocks_2": 4,
    "nr_residual_blocks_3": 2,
    "res_branches_1": 1,
    "res_branches_2": 1,
    "res_branches_3": 4,
    "widen_factor_1": 6.241141,
    "widen_factor_2": 1.388867,
    "widen_factor_3": 3.344766,
    "auxiliary": True}

test_config = stupid_dict(test_config)

if __name__=="__main__":
    from resnet import ResNet
    import torch
    import torch.nn as nn

    test_data = torch.zeros((10,3,32,32))
    test_data_out = torch.empty(10, dtype=torch.long).random_(10)
    #test_data = torch.zeros((10,3,64,64)).to("cuda:0")

    test_resnet = ResNet(config=test_config,
                         in_features=(3,32,32),
                         out_features=10,
                         final_activation=None)

    #criterion = CrossEntropyLoss()
    #optimizer = optim.SGD(test_resnet.parameters(), lr=0.01, momentum=0.9)
    #optimizer.zero_grad()
    preds = test_resnet(test_data)
    print("Test predicitons", preds)
    #print("Test predicitons", preds[0])
    #print("Test predicitons", test_resnet.model.forward(test_data))

    #loss = criterion(preds, test_data_out)
    #loss = criterion(preds[0], test_data_out)
    #loss.backward()
    #optimizer.step()

    

    print(test_resnet)

    #print(test_resnet.model[0:4])
    #print("Test predicitons at auxiliary", test_resnet.model[0:4].forward(test_data).detach().numpy().shape)
