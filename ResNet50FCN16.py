import torch
import torch.nn as nn
import torchvision.models as models


class ResNet50FCN16(nn.Module):
    def __init__(self, num_classes=15, pretrained=True):
        super(ResNet50FCN16, self).__init__()
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        # downsampling layers
        print(resnet.fc.in_features)
        # stage 1 of ResNet50
        self.stage1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)

        # Stage 2 to 5 corresponding to convolutional blocks 2-5 in ResNet50 architecture
        self.stage2 = resnet.layer1
        self.stage3 = resnet.layer2
        self.stage4 = resnet.layer3
        self.stage5 = resnet.layer4  # final layer excluding the average pooling layer and the FC1000 layer

        # use two convolutional layers to replace FC layer, intermediate neuron number set to 1000 to maintain the capacity of learning pattern as to FC1000.
        self.conv6 = nn.Conv2d(2048, 1000, kernel_size=16)
        self.relu6 = nn.ReLU(inplace=True)
        self.drop6 = nn.Dropout2d()

        self.conv7 = nn.Conv2d(1000, 1000, kernel_size=1)
        self.relu7 = nn.ReLU(inplace=True)
        self.drop7 = nn.Dropout2d()

        self.final = nn.Conv2d(1000, num_classes, kernel_size=1)

        # Upsampling layers
        self.upsample2x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=2, stride=2, bias=False)
        self.upsample4x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=4, stride=4, bias=False)
        # self.upsample8x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=8, stride=8, bias=False)
        self.upsample16x = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=16, stride=16, bias=False)

        # Skip connection from stage4(corresponding to pool4 in the vgg16FCN16 paper)
        self.layer3_avgpool = nn.AvgPool2d(16, stride=16)  # Add an average pooling layer to downsample the
        # output tensor size of convolution block7 into (2,2), thus match the input dimension of 2x upsampled final
        # layer output.
        self.layer3_conv = nn.Conv2d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        # Encoder forward pass
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        layer4 = self.stage4(x)
        x = self.stage5(layer4)

        # Fully connected layer replacement
        x = self.relu6(self.conv6(x))
        x = self.drop6(x)
        x = self.relu7(self.conv7(x))
        x = self.drop7(x)
        x = self.final(x)

        # Upsample by 2
        x = self.upsample2x(x)

        # Skip connection of the 2x upsampled final layer with the average pooled prediction of layer4
        pool4 = self.layer3_avgpool(layer4)
        pool4_pred = self.layer3_conv(pool4)
        x = x + pool4_pred
        # upsampling by 4
        x = self.upsample4x(x)
        # upsampling by 4
        x = self.upsample4x(x)
        # upsampling by 16
        x = self.upsample16x(x)

        return x


model = ResNet50FCN16()

input_tensor = torch.randn(1, 3, 512, 512)  # Example input tensor
output = model(input_tensor)
print(output.shape)  # Should match the input size
