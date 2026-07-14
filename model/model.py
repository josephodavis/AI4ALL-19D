import torch
import torch.nn as nn
import torch.nn.functional as F

class FirstCNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Convolution Layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        
        # Batch Normalization Layers
        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm2d(512)
        
        # Pooling Layers
        self.pool = nn.MaxPool2d(2, 2)          
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1)) 
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256) # stabilize the dense layer
        self.dropout = nn.Dropout(0.3) # helps prevent memory overfitting          
        self.fc2 = nn.Linear(256, 5)

    def forward(self, x):
        # Conv -> BN -> ReLU -> MaxPool
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Input 224x224 -> Outputs 112x112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Outputs 56x56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Outputs 28x28
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  # Outputs 14x14
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  # Outputs 7x7
        
        # collapse the spatial dimensions to a fixed size (1x1) for the dense layers
        x = self.adaptive_pool(x)                       
        
        # Flatten and process dense features
        x = torch.flatten(x, 1)                        
        
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)                                
        return x

# class FirstCNN(nn.Module):
#     def __init__(self):
#         super().__init__()

#         # Block 1
#         self.block1 = nn.Sequential(
#             nn.Conv2d(3, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(32, 32, kernel_size=3, padding=1),
#             nn.BatchNorm2d(32),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2, 2)
#         )

#         # Block 2
#         self.block2 = nn.Sequential(
#             nn.Conv2d(32, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(64, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2, 2)
#         )

#         # Block 3
#         self.block3 = nn.Sequential(
#             nn.Conv2d(64, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(128, 128, kernel_size=3, padding=1),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2, 2)
#         )

#         # Block 4
#         self.block4 = nn.Sequential(
#             nn.Conv2d(128, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(256, 256, kernel_size=3, padding=1),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2, 2)
#         )

#         # Block 5
#         self.block5 = nn.Sequential(
#             nn.Conv2d(256, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),

#             nn.Conv2d(512, 512, kernel_size=3, padding=1),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),

#             nn.MaxPool2d(2, 2)
#         )

#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

#         self.classifier = nn.Sequential(
#             nn.Linear(512, 256),
#             nn.BatchNorm1d(256),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.3),
#             nn.Linear(256, 5)
#         )

#     def forward(self, x):
#         x = self.block1(x)   # 224 -> 112
#         x = self.block2(x)   # 112 -> 56
#         x = self.block3(x)   # 56 -> 28
#         x = self.block4(x)   # 28 -> 14
#         x = self.block5(x)   # 14 -> 7

#         x = self.adaptive_pool(x)
#         x = torch.flatten(x, 1)

#         x = self.classifier(x)

#         return x