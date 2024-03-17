import torch
import torch.nn as nn
import torch.nn.functional as F

class Prev_Net(nn.Module):
    def __init__(self):
        super(Prev_Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3, 
                               kernel_size = 5, stride = 1, padding = 2,
                               bias = True)
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3,
                               kernel_size = 3, stride = 4, padding = 0,
                               bias = True)
        
        self.fc1 = nn.Linear(60, 23)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = nn.Flatten()(x)
        x = self.fc1(x)
        
        return x

# Mine that followed by Sun's Net
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3,
                               kernel_size = 8, stride = 1, padding = 2,
                               bias = True)
        self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3,
                               kernel_size = 3, stride = 4, padding = 0,
                               bias = True)
        
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(60, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 23)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = nn.Flatten()(x)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x

# # Mine that followed by Sun's Net
# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
        
#         self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 3,
#                                kernel_size = 5, stride = 1,
#                                padding = 0, bias = True)
#         self.conv2 = nn.Conv2d(in_channels = 3, out_channels = 3,
#                                kernel_size = 1, stride = 1,
#                                padding = 0, bias = True)
        
#         self.fc1 = nn.Linear(432, 64)
#         self.fc2 = nn.Linear(64, 32)
#         self.fc3 = nn.Linear(32, 23)
        
#         self.relu = nn.ReLU()
        
#     def forward(self, x):
#         x = self.relu(self.conv1(x))
#         x = self.relu(self.conv2(x))
#         x = nn.Flatten()(x)
        
#         x = self.relu(self.fc1(x))
#         x = self.relu(self.fc2(x))
#         x = self.fc3(x)

#         return x

class VGGNet16(nn.Module):
    def __init__(self, num_classes = 23):
        super(VGGNet16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (10, 2), stride=1),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (10, 2), stride=1),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (10, 2), stride=1),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size = (10, 2), stride=1),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # Adjust output size based on your needs
        self.classifier = nn.Sequential(
            nn.Linear(512 * 4 * 4, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.downsample(residual)
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 3, kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(BasicBlock, 3, 3, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 3, 3, stride=4)

        self.fc1 = nn.Linear(12, 48)
        self.fc2 = nn.Linear(48, 32)
        self.fc3 = nn.Linear(32, 23)

    def _make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = x.view(x.size(0), -1)

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        return x