import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

class ResNet18Classifier(nn.Module):
    def __init__(self, n_classes=6, n_neurons=200):
        super(ResNet18Classifier, self).__init__()
        self.n_classes = n_classes
        self.n_neurons = n_neurons

        resnet18 = hub.load(
            'pytorch/vision:v0.10.0',
            'resnet18',
            pretrained=True
        )

        resnet18.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=self.n_neurons, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_neurons, out_features=self.n_classes, bias=True)
        )

        self.resnet18 = resnet18

    def forward(self, x):
        out = self.resnet18(x)
        return out

class ResNet34Classifier(nn.Module):
    def __init__(self, n_classes=6, n_neurons=200):
        super(ResNet34Classifier, self).__init__()
        self.n_classes = n_classes
        self.n_neurons = n_neurons

        resnet34 = hub.load(
            'pytorch/vision:v0.10.0',
            'resnet34',
            pretrained=True
        )

        resnet34.fc = nn.Sequential(
            nn.Linear(in_features=512, out_features=self.n_neurons, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_neurons, out_features=self.n_classes, bias=True)
        )

        self.resnet34 = resnet34

    def forward(self, x):
        out = self.resnet34(x)
        return out

class ResNet50Classifier(nn.Module):
    def __init__(self, n_classes=6, n_neurons=200):
        super(ResNet50Classifier, self).__init__()
        self.n_classes = n_classes
        self.n_neurons = n_neurons

        resnet50 = hub.load(
            'pytorch/vision:v0.10.0',
            'resnet50',
            pretrained=True
        )

        resnet50.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1000, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=1000, out_features=self.n_neurons, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.n_neurons, out_features=self.n_classes, bias=True)
        )

        self.resnet50 = resnet50

    def forward(self, x):
        out = self.resnet50(x)
        return out