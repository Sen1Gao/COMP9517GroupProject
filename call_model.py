# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:20:54 2024

@author: Sen
"""
import gc
import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision import transforms
import matplotlib.pyplot as plt

palette=[(0, 0, 0),(60, 180, 75),(255, 225, 25),(0, 130, 200),
         (145, 30, 180),(70, 240, 240),(240, 50, 230),(210, 245, 60),
         (230, 25, 75),(0, 128, 128),(170, 110, 40),(170, 255, 195),
         (128, 128, 0),(250, 190, 190),(0, 0, 128),(128, 128, 128)]

def load_image_and_label(image_path,label_path,target_size):
    image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size,interpolation=cv2.INTER_NEAREST)
    label=cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
    label = cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
    return image,label

def render_image(label):
    label_color=np.zeros((label.shape[0],label.shape[1],3),dtype=np.uint8)
    for row in range(label.shape[0]):
        for col in range(label.shape[1]):
            label_color[row][col][0]=palette[label[row][col]][0]
            label_color[row][col][1]=palette[label[row][col]][1]
            label_color[row][col][2]=palette[label[row][col]][2]
    return label_color

def calculate_IoU(pred_label, label, class_index):
    mask_pred_label = (pred_label == class_index)
    mask_label = (label == class_index)
    intersection = (mask_pred_label & mask_label).sum()
    union = (mask_pred_label | mask_label).sum()
    iou = intersection/union if union > 0 else 0
    return iou


def calculate_mIoU(pred_label, label, class_num):
    ious = []
    for index in range(class_num):
        iou = calculate_IoU(pred_label, label, index)
        ious.append(iou)
    return np.mean(np.array(ious)), ious

class UNet(nn.Module):
    def __init__(self, class_num):
        super(UNet, self).__init__()

        def CBR(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        self.encoder1 = CBR(3, 32)
        self.encoder2 = CBR(32, 64)
        self.encoder3 = CBR(64, 128)
        self.encoder4 = CBR(128, 256)

        self.pool = nn.MaxPool2d(2)

        self.bottleneck = CBR(256, 512)

        self.upconv4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder4 = CBR(512, 256)

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder3 = CBR(256, 128)

        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder2 = CBR(128, 64)

        self.upconv1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.decoder1 = CBR(64, 32)

        self.conv_last = nn.Conv2d(32, class_num, kernel_size=1)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool(enc1))
        enc3 = self.encoder3(self.pool(enc2))
        enc4 = self.encoder4(self.pool(enc3))

        bottleneck = self.bottleneck(self.pool(enc4))

        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat((dec4, enc4), dim=1)
        dec4 = self.decoder4(dec4)

        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)

        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)

        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)

        return self.conv_last(dec1)
    
transform = transforms.Compose([
    transforms.ToTensor()
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model_folder_path="C:/Users/JARVIS/Documents/Projects/CustomWildScenes2d"
model_name="best_model.pth"

image_path="C:/Users/JARVIS/Documents/Projects/CustomWildScenes2d/test/image/1624329385-569874078.png"
label_path=image_path.replace("image", "newIndexLabel")
target_size=(512,512)
class_num=16
try:
    model = torch.load(f"{model_folder_path}/{model_name}")
    model.eval()
    with torch.no_grad():
        image,label=load_image_and_label(image_path,label_path,target_size)
        image_tensor=transform(image).unsqueeze(0)
        label_tensor=torch.tensor(label, dtype=torch.long)
        output = model(image_tensor.to(device))
        prob = torch.softmax(output, dim=1)
        pred = torch.argmax(prob, dim=1)
        pred_np = pred.cpu().numpy().astype(np.uint8)
        miou,ious=calculate_mIoU(pred_np[0],label,class_num)
        print(f"mIoU:{miou}")
        print(f"IoUs:{ious}")
        label_color=render_image(label)
        pred_color=render_image(pred_np[0])
        
        fig = plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image)
        plt.title("Image")
        plt.axis('off')
        plt.subplot(1, 3, 2)
        plt.imshow(label_color)
        plt.title("Ground Truth")
        plt.axis('off')
        plt.subplot(1, 3, 3)
        plt.imshow(pred_color)
        plt.title("Prediction")
        plt.axis('off')
        plt.show()
except:
    print("Catch exception. Cleaning up...")
finally:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    del model
    gc.collect()
    print("Cleanup complete. Exiting...")