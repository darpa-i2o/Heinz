import sys
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from .modules.quantize import quantize, quantize_grad, QConv2d, QLinear, RangeBN
__all__ = ['densenet_quantized']

NUM_BITS = 8
NUM_BITS_WEIGHT = 8
NUM_BITS_GRAD = None
BIPRECISION = False


# __all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']


model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def conv3x3(in_planes, out_planes, stride=1, pool_size = None):
    "3x3 convolution with padding"
    return QConv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                   padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION, pool_size = pool_size)


def densenet121(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet169(pretrained=False, **kwargs):
    r"""Densenet-169 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 32, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet169'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet201(pretrained=False, **kwargs):
    r"""Densenet-201 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet201'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model


def densenet161(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = DenseNet(num_init_features=96, growth_rate=48, block_config=(6, 12, 36, 24),
                     **kwargs)
    if pretrained:
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet161'])
        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]
        model.load_state_dict(state_dict)
    return model

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, pool_size):
        super(_DenseLayer, self).__init__()
        self.dense_module = nn.Sequential(OrderedDict([('norm1', nn.BatchNorm2d(num_input_features)),
                      ('relu1', nn.ReLU(inplace=True)),
                      ('conv1', QConv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False,num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,                    num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION, pool_size=pool_size, downsample=True) ),
                      ('norm2', RangeBN(bn_size * growth_rate, num_bits=NUM_BITS,
                                       num_bits_grad=NUM_BITS_GRAD)),
                      ('relu2', nn.ReLU(inplace=True)),
                      ('conv2', QConv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT,                    num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION, pool_size=pool_size, downsample=True))])
                       )
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = self.dense_module(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)




class _DenseBlock(nn.Module):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, pool_size):
        super(_DenseBlock, self).__init__()
        self.layers = []
        self.num_layers = num_layers
        for i in range(num_layers):
            setattr(self, 'denselayer%s' % i, self._make_layer(i, i+1, num_input_features, growth_rate, bn_size, drop_rate, pool_size))

    def _make_layer(self, front_layer_idx, back_layer_index, num_input_features, growth_rate, bn_size, drop_rate, pool_size):
        modules = []
        for i in range(front_layer_idx, back_layer_index):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate, bn_size, drop_rate, pool_size)
            modules.extend([layer])
        return nn.Sequential(*modules)

    def forward(self, x):
        for i in range(self.num_layers):
            x = getattr(self, 'denselayer{}'.format(i))(x)
        return x
       
class _Transition(nn.Module):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.trans_module = nn.Sequential(OrderedDict([('norm', nn.BatchNorm2d(num_input_features)),
                       ('relu', nn.ReLU(inplace=True)),
                       ('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)),
                       ('pool', nn.AvgPool2d(kernel_size=2, stride=2))]))
    def forward(self, x):
        return self.trans_module(x)



class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
    """

    def __init__(self, growth_rate=12, block_config=(6, 12, 24, 16),
                 num_init_features=24, bn_size=4, drop_rate=0, num_classes=10):

        super(DenseNet, self).__init__()

        # First convolution
        self.base_layer = QConv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False, num_bits=NUM_BITS, num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD,biprecision=BIPRECISION, pool_size=32)

        num_features = num_init_features


        # denseblock 0
        self.DenseBlock0 = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, pool_size=32)
        num_features = (num_features + block_config[0] * growth_rate)
        self.trans0 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2
 

        # denseblock 1
        self.DenseBlock1 = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, pool_size=16)
        num_features = (num_features + block_config[1] * growth_rate)
        self.trans1 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2      



        # denseblock 2
        self.DenseBlock2 = _DenseBlock(num_layers=block_config[2], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, pool_size=8)
        num_features = (num_features + block_config[2] * growth_rate)
        self.trans2 = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2


        # denseblock 3
        self.DenseBlock3 = _DenseBlock(num_layers=block_config[3], num_input_features=num_features,
                                bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate, pool_size=4)
        num_features = (num_features + block_config[3] * growth_rate)


        # Final batch norm
        self.bn_norm = RangeBN(num_features, num_bits=NUM_BITS,
                               num_bits_grad=NUM_BITS_GRAD)

        # Linear layer
        self.classifier = QLinear(num_features, num_classes, num_bits=NUM_BITS,
                          num_bits_weight=NUM_BITS_WEIGHT, num_bits_grad=NUM_BITS_GRAD, biprecision=BIPRECISION)
        
        self.regime = [
            {'epoch': 0, 'optimizer': 'SGD', 'lr': 1e-1,
             'weight_decay': 1e-4, 'momentum': 0.9},
            {'epoch': 81, 'lr': 1e-2},
            {'epoch': 122, 'lr': 1e-3, 'weight_decay': 0},
            {'epoch': 164, 'lr': 1e-4}
        ]

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.base_layer(x)
        features = self.DenseBlock0(features)
        features = self.trans0(features)
        features = self.DenseBlock1(features)
        features = self.trans1(features)
        features = self.DenseBlock2(features)
        features = self.trans2(features)
        features = self.DenseBlock3(features)
        features = self.bn_norm(features)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=4, stride=1).view(features.size(0), -1)
        out = self.classifier(out)
        return out
    
    
def densenet_quantized(**kwargs):
    num_classes, depth, dataset = map(
        kwargs.get, ['num_classes', 'depth', 'dataset'])
    
    if dataset == 'cifar10':
        if depth == 121:
            return DenseNet(num_init_features=24, growth_rate=12, block_config=(6, 12, 24, 16),
                     )
        
        elif depth == 169:
            return DenseNet(num_init_features=24, growth_rate=12, block_config=(6, 12, 32, 32),
                     )
        
        elif depth == 201:
            return DenseNet(num_init_features=24, growth_rate=12, block_config=(6, 12, 48, 32),
                     )
        
        elif depth == 161:
            return DenseNet(num_init_features=24, growth_rate=12, block_config=(6, 12, 36, 24),
                     )
            
