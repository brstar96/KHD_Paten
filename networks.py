from torchvision import models
from backbones.octConv_ResNet import oct_resnet50, oct_resnet101, oct_resnet152
from backbones.SENet import senet154, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
from backbones.efficientnet import EfficientNet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # False = 역전파 중 해당 Tensor에 대한 Gradient를 계산하지 않을 것임을 의미
            param.requires_grad = False

def initialize_model(model_name, use_pretrained=True):
    # These models are pretrained via ImageNet-1000 class
    if model_name == "resnet101":
        return models.resnet101(pretrained=use_pretrained, progress=True)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=use_pretrained, progress=True)

    # Constructs a Octave ResNet-152 model.(pretrained (bool): If True, returns a model pre-trained on ImageNet)
    elif model_name == "oct_resnet50":
        return oct_resnet50(pretrained=use_pretrained, progress=True)
    elif model_name == "oct_resnet101":
        return oct_resnet101(pretrained=use_pretrained, progress=True)
    elif model_name == "oct_resnet152":
        return oct_resnet152(pretrained=use_pretrained, progress=True)

    # 아래 5개의 모델은 pretrained=None이면 전이학습을 하지 않음.
    elif model_name == "senet154":
        return senet154(num_classes=1000, pretrained='imagenet')
    elif model_name == "se_resnet101":
        return se_resnet101(num_classes=1000, pretrained='imagenet')
    elif model_name == "se_resnet152":
        return se_resnet152(num_classes=1000, pretrained='imagenet')
    elif model_name == 'se_resnext50_32x4d':
        return se_resnext50_32x4d(num_classes=1000, pretrained='imagenet')
    elif model_name == 'se_resnext101_32x4d':
        return se_resnext101_32x4d(num_classes=1000, pretrained='imagenet')
    elif model_name == "resnext50_32x4d":
        # Aggregated Residual Transformation for Deep Neural Networks<https://arxiv.org/pdf/1611.05431.pdf>`
        return models.resnext50_32x4d(pretrained=use_pretrained, progress=True)
    elif model_name == "resnext101_32x8d":
        return models.resnext101_32x8d(pretrained=use_pretrained, progress=True)

    # ImageNet pretrained efficientnet-b3, b4
    elif model_name == 'efficientnetb3':
        return EfficientNet.from_pretrained('efficientnet-b3')
    elif model_name == 'efficientnetb4':
        return EfficientNet.from_pretrained('efficientnet-b4')
    else:
        print("Wrong define model parameter input.")
        raise ValueError