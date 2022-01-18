from torch import nn
from torchvision import models


def grl_hook(coeff):

    def fun1(grad):
        return -coeff * grad.clone()

    return fun1


def create_resnet50_features():
    model_resnet = models.resnet50(pretrained=True)
    conv1 = model_resnet.conv1
    bn1 = model_resnet.bn1
    relu = model_resnet.relu
    maxpool = model_resnet.maxpool
    layer1 = model_resnet.layer1
    layer2 = model_resnet.layer2
    layer3 = model_resnet.layer3
    layer4 = model_resnet.layer4
    avgpool = model_resnet.avgpool
    feature_layers = nn.Sequential(conv1, bn1, relu, maxpool, \
                            layer1, layer2, layer3, layer4, avgpool)
    return feature_layers


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find(
            'ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
