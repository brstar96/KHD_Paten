from torchvision import models
from backbones.octConv_ResNet import oct_resnet26, oct_resnet50, oct_resnet101, oct_resnet152
from backbones.SENet import senet154, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
from backbones.efficientnet import EfficientNet

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            # False = 역전파 중 해당 Tensor에 대한 Gradient를 계산하지 않을 것임을 의미
            param.requires_grad = False

# If use_pretrained=True, below models are using ImageNet pretrained weights.
def initialize_model(model_name, use_pretrained=False, input_channels=None, num_classes=None):
    # If use_pretrained = True, These models are pretrained via ImageNet-1000 class
    if model_name == "resnet101":
        return models.resnet101(pretrained=use_pretrained, progress=True)
    elif model_name == "resnet152":
        return models.resnet152(pretrained=use_pretrained, progress=True)

    # Constructs a Octave ResNet-152 model.(pretrained (bool): If True, returns a model pre-trained on ImageNet)
    elif model_name == "oct_resnet26":
        return oct_resnet26(input_channels=input_channels, num_classes=num_classes)
    elif model_name == "oct_resnet50":
        return oct_resnet50(input_channels=input_channels, num_classes=num_classes)
    elif model_name == "oct_resnet101":
        return oct_resnet101(input_channels=input_channels, num_classes=num_classes)
    elif model_name == "oct_resnet152":
        return oct_resnet152(input_channels=input_channels, num_classes=num_classes)

    # 아래 5개의 모델은 pretrained=None이면 전이학습을 하지 않음.
    elif model_name == "senet154":
        if use_pretrained == False:
            return senet154(num_classes=num_classes, pretrained=None)
        else:
            return senet154(num_classes=num_classes, pretrained='imagenet')
    elif model_name == "se_resnet101":
        if use_pretrained == False:
            return se_resnet101(num_classes=num_classes, pretrained=None)
        else:
            return se_resnet101(num_classes=num_classes, pretrained='imagenet')
    elif model_name == "se_resnet152":
        if use_pretrained == False:
            return se_resnet152(num_classes=num_classes, pretrained=None)
        else:
            return se_resnet152(num_classes=num_classes, pretrained='imagenet')
    elif model_name == 'se_resnext50_32x4d':
        if use_pretrained == False:
            return se_resnext50_32x4d(num_classes=num_classes, pretrained=None)
        else:
            return se_resnext50_32x4d(num_classes=num_classes, pretrained='imagenet')
    elif model_name == 'se_resnext101_32x4d':
        if use_pretrained == False:
            return se_resnext101_32x4d(num_classes=num_classes, pretrained=None)
        else:
            return se_resnext101_32x4d(num_classes=num_classes, pretrained='imagenet')
    elif model_name == "resnext50_32x4d":
        # Aggregated Residual Transformation for Deep Neural Networks<https://arxiv.org/pdf/1611.05431.pdf>`
        # If progress=True, print pretrained model downloading status.
        return models.resnext50_32x4d(pretrained=use_pretrained, progress=True)
    elif model_name == "resnext101_32x8d":
        return models.resnext101_32x8d(pretrained=use_pretrained, progress=True)

    # ImageNet pretrained efficientnet-b3, b4
    elif model_name == 'efficientnetb3':
        return EfficientNet.from_scratch(model_name='efficientnet-b3', num_classes=num_classes)
    elif model_name == 'efficientnetb4':
        return EfficientNet.from_scratch(model_name='efficientnet-b4', num_classes=num_classes)
    elif model_name == 'efficientnetb5':
        return EfficientNet.from_scratch(model_name='efficientnet-b5', num_classes=num_classes)
    else:
        print("Wrong define model parameter input.")
        raise ValueError