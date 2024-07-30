import torch.nn as nn
import torch.nn.functional as F
import torch.hub as hub

class DenseNetClassifier(nn.Module):
    def __init__(self, n_classes=6, n_neurons=64, dropout=0.2):
        super().__init__()

        self.n_classes = n_classes
        self.n_neurons = n_neurons
        self.dropout = dropout

        # Get pretrained denseNet201 for feature extraction.
        densenet = hub.load(
            'pytorch/vision:v0.10.0',
            'densenet201',
            pretrained=True
        )
        self.feature_extractor = nn.Sequential(
            densenet.features,
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)), # Same as global pooling
        )

        self.batchNorm1 = nn.BatchNorm1d(1920)

        self.fc_block1 = nn.Sequential(
            nn.Linear(1920, self.n_neurons),
            nn.BatchNorm1d(self.n_neurons),
            nn.LeakyReLU(inplace=True)
        )

        self.fc_block2 = nn.Sequential(
            nn.Linear(self.n_neurons, self.n_neurons),
            nn.BatchNorm1d(self.n_neurons),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Linear(self.n_neurons, self.n_classes)

    def forward(self, x, is_train=True):
        assert x.shape[1:] == (3, 224, 224), \
            f"Input shape {x.shape[1:]} not as desired!"

        # in - (b, 3, 224, 224), out - (b, 1920, 1, 1)
        out = self.feature_extractor(x)

        # Flattening output to (b, 1920)
        out = out.reshape(x.shape[0], -1)

        # in - (b, 1920), out - (b, 1920)
        out = self.batchNorm1(out)

        # in - (b, 1920), out - (b, 128)
        out = self.fc_block1(out)
        out = F.dropout(out, p=self.dropout, training=is_train)

        # in - (b, 128), out - (b, 128)
        out = self.fc_block2(out)

        # in - (b, 128), out - (b, 4)
        out = self.classifier(out)

        return out