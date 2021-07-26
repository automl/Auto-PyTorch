__all__ = ['EfficientNetB0', 'EfficientNetB1', 'EfficientNetB2', 'EfficientNetB3', 'EfficientNetB4', 'EfficientNetB5', 'EfficientNetB6', 'EfficientNetB7']

import torch
from autoPyTorch.components.networks.base_net import BaseImageNet
from timm.models import (efficientnet_b0,
                         efficientnet_b1,
                         efficientnet_b2,
                         efficientnet_b3,
                         efficientnet_b4,
                         efficientnet_b5,
                         efficientnet_b6,
                         efficientnet_b7)


class EfficientNetB0(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB0, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b0(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.2, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b0(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.2, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB1(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB1, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b1(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.2, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b1(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.2, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB2(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB2, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b2(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.3, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b2(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.3, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB3(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB3, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b3(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.3, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b3(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.3, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB4(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB4, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b4(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.4, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b4(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.4, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB5(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB5, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b5(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.4, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b5(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.4, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB6(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB6, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b6(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.5, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b6(pretrained=False, num_classes=out_features, in_chans=in_features[0], drop_rate=0.5, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x


class EfficientNetB7(BaseImageNet):


    def __init__(self, config, in_features, out_features, final_activation, **kwargs):
        super(EfficientNetB7, self).__init__(config, in_features, out_features, final_activation)

        if len(in_features)==2:
            self.net = efficientnet_b7(pretrained=False, num_classes=out_features, in_chans=1, drop_rate=0.5, drop_connect_rate=0.2, bn_momentum=0.01)
        if len(in_features)==3:
            self.net = efficientnet_b7(pretrained=True, num_classes=out_features, in_chans=in_features[0], drop_rate=0.5, drop_connect_rate=0.2, bn_momentum=0.01)
        
        self.final_activation = final_activation


    def forward(self, x):
        x = self.net.forward(x)
    
        if not self.training and self.final_activation is not None:
            x = self.final_activation(x)
        return x
