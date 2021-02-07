from deepsentinel.models.encoders.basic_bottleneck import BasicBottleneck
from deepsentinel.models.encoders.resnet import resnet18, resnet34, resnet50

encoders = {
    'basic_bottleneck':BasicBottleneck,
    'resnet18':resnet18,
    'resnet34':resnet34,
    'resnet50':resnet50,
}