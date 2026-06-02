import torch
import torchvision.models as models

baseline_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

for param in baseline_model.parameters():
    param.requires_grad = True

num_classes = 2
in_features = baseline_model.fc.in_features

baseline_model.fc = torch.nn.Sequential(
    torch.nn.Dropout(p=0.2),
    torch.nn.Linear(in_features, num_classes),
)
