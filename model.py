from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn


NUM_OUTPUTS = 2


class ResNetCustom(nn.Module):
    def __init__(self):
      super().__init__()
      self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
      self.last_layer = nn.Linear(self.model.fc.in_features, NUM_OUTPUTS)
      self.model.fc = nn.Identity()

    def forward(self, x):
      return self.last_layer(self.model(x))

model = ResNetCustom()