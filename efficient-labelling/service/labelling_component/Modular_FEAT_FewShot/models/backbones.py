import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18FeatEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout_rate=0.3):
        super().__init__()
        base = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.encoder = nn.Sequential(*list(base.children())[:-1])  # Exclude final FC layer
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout_rate)
        self.projection = nn.Linear(512, hidden_dim)
#         nn.init.kaiming_normal_(self.projection.weight, nonlinearity='relu')
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        if self.projection.bias is not None:
                nn.init.constant_(self.projection.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.dropout(x)
        return self.projection(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )

    def forward(self, x):
        return self.block(x)

class ConvNet4(nn.Module):
    def __init__(self, in_channels=3, feat_dim=640):
        super().__init__()
        self.encoder_conv = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 160),
            ConvBlock(160, 320),
            ConvBlock(320, 320)
        )
        self.flatten = nn.Flatten()
        spatial_output_size = 5 
        conv_output_dim = 320 * spatial_output_size * spatial_output_size # 320 * 25 = 8000
        self.projection = nn.Linear(conv_output_dim, feat_dim)
        nn.init.kaiming_normal_(self.projection.weight, mode='fan_out', nonlinearity='relu')
        if self.projection.bias is not None:
            nn.init.constant_(self.projection.bias, 0)
        

    def forward(self, x):
        x = self.encoder_conv(x)
        x = self.flatten(x)
        x = self.projection(x) 
        return x

class ResNet12(nn.Module):
    def __init__(self, in_channels=3, feat_dim=640):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 160, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(160, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(160, 320, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(320, momentum=0.1, eps=1e-5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(320, feat_dim, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d(1)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)

def get_backbone(name, hidden_dim, dropout_rate=0.3):
    if name == 'ResNet18':
        model= ResNet18FeatEncoder(hidden_dim, dropout_rate)
    elif name == 'ConvNet4':
        model= ConvNet4(feat_dim=hidden_dim)
    elif name == 'ResNet12':
        model= ResNet12(feat_dim=hidden_dim)
    else:
        raise ValueError(f"Unknown backbone: {name}")
    if name != 'ResNet18': # Apply this block only if not ResNet18
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear): # This will catch ConvNet4's new projection layer
                nn.init.kaiming_normal_(
                    m.weight, 
                    mode='fan_out', 
                    nonlinearity='relu'
                )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            
    return model
