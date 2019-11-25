import torch.nn as nn
from torchvision import models
import pyramidnet as PYRM

class BaseNetwork(nn.Module):
    """ Load Pretrained Module """

    def __init__(self, model_name, embedding_dim, feature_extracting, use_pretrained):
        super(BaseNetwork, self).__init__()
        self.model_name = model_name
        self.embedding_dim = embedding_dim
        self.feature_extracting = feature_extracting
        self.use_pretrained = use_pretrained

        self.model_ft = initialize_model(self.model_name,
                                         self.embedding_dim,
                                         self.feature_extracting,
                                         self.use_pretrained)
    def forward(self, x):
        out = self.model_ft(x)
        return out

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # False = 역전파 중 해당 Tensor에 대한 Gradient를 계산하지 않을 것임을 의미
            param.requires_grad = False

def initialize_model(model_name, embedding_dim, feature_extracting, use_pretrained=True):
    if model_name == "densenet161":
        model_ft = models.densenet161(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_features, embedding_dim)
    elif model_name == "resnet101":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, embedding_dim)
    elif model_name == "inceptionv3":
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_features, embedding_dim)
    elif model_name == "seresnext":
        model_ft = se_resnext101_32x4d(num_classes=1000)
        set_parameter_requires_grad(model_ft, feature_extracting)
        num_features = model_ft.last_linear.in_features
        model_ft.last_linear = nn.Linear(num_features, embedding_dim)
    elif model_name == 'pyramidnet-200':
        model = PYRM.PyramidNet(dataset, depth, alpha, numberofclass, bottleneck)
        set_parameter_requires_grad(model, feature_extracting)
        num_features = model.last_linear.in_features
        model.last_linear = nn.Linear(num_features, embedding_dim)
    else:
        raise ValueError

    return model_ft