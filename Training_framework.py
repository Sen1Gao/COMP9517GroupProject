# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 00:09:38 2024

@author: Sen
"""

import gc
from pathlib import Path
from tqdm import tqdm
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, dataset_folder_path, csv_file_type, loading_mode, target_size=None, transform=None):
        self.dataset_folder_path = dataset_folder_path
        self.csv_file_type = csv_file_type
        self.sampled_image_path_list = np.load(
            f"{dataset_folder_path}/sampled_{csv_file_type}_image_path_list.npy")
        self.loading_mode = loading_mode
        self.target_size = target_size
        self.transform = transform
        self.image_list = []
        self.label_list = []
        if self.loading_mode == 'pre':
            self.pre_load_and_process()
        elif self.loading_mode == 'real':
            pass
        else:
            raise Exception(
                "You must assign the value of loading_mode,'pre' or 'real'")

    def process(self, image_name):
        image_path = f"{self.dataset_folder_path}/{self.csv_file_type}/image/{image_name}"
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label_name = image_name
        label_path = f"{self.dataset_folder_path}/{self.csv_file_type}/newIndexLabel/{image_name}"
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size,
                               interpolation=cv2.INTER_NEAREST)
            label = cv2.resize(label, self.target_size,
                               interpolation=cv2.INTER_NEAREST)
        return image, label

    def pre_load_and_process(self):
        print(f"Start loading {self.csv_file_type} data")
        for i in self.sampled_image_path_list:
            image_name = Path(i).name
            image, label = self.process(image_name)
            self.image_list.append(image)
            self.label_list.append(label)
        print(f"Loading {self.csv_file_type} data has been completed")

    def __len__(self):
        return len(self.sampled_image_path_list)

    def __getitem__(self, idx):
        image = None
        label = None
        if self.loading_mode == 'pre':
            image = self.image_list[idx]
            label = self.label_list[idx]
        if self.loading_mode == 'real':
            image_name = Path(self.sampled_image_path_list[idx]).name
            image, label = self.process(image_name)
        if self.transform is not None:
            image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label


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
                nn.ReLU(inplace=True),
                # nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                # nn.BatchNorm2d(out_channels),
                # nn.ReLU(inplace=True)
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

dataset_folder_path = "C:/Users/JARVIS/Documents/Projects/CustomWildScenes2d"

loading_mode = 'pre'
target_size = (512, 512)

train_data = CustomDataset(dataset_folder_path, 'train', loading_mode, target_size, transform)
val_data = CustomDataset(dataset_folder_path, 'val',loading_mode, target_size, transform)
test_data = CustomDataset(dataset_folder_path, 'test',loading_mode, target_size, transform)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

class_num = 16
model = UNet(class_num).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
is_StepLR = False
scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

batch_size = 6
train_loader = DataLoader(train_data, batch_size, shuffle=False)
val_loader = DataLoader(val_data, batch_size, shuffle=False)
test_loader = DataLoader(test_data, batch_size, shuffle=False)

epochs = 50
x = np.arange(0, epochs, 1)
y_training_loss = np.array([0]*epochs, dtype=float)
y_val_loss = np.array([0]*epochs, dtype=float)
y_avg_miou = np.array([0]*epochs, dtype=float)

is_early_stopping = True
best_avg_miou = float('inf')
early_stopping_patience = 10
no_improvement_epochs = 0

try:
    for epoch in range(epochs):
        message = f"Current epoch:{epoch+1}/{epochs} Training:"
        model.train()
        training_loss_sum = 0.0
        for images, labels in tqdm(train_loader, desc=message, leave=False):
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            training_loss_sum += loss.item()
        training_loss_avg = training_loss_sum / len(train_loader)
        y_training_loss[epoch] = training_loss_avg

        message = f"Current epoch:{epoch+1}/{epochs} validating:"
        model.eval()
        val_loss_sum = 0.0
        miou_sum = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=message, leave=False):
                images = images.to(device)
                labels = labels.to(device).long()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_sum += loss.item()
                pred_labels = torch.argmax(outputs, dim=1)
                pred_labels_np = pred_labels.cpu().numpy()
                labels_np = labels.cpu().numpy()
                batch_miou_sum = 0.0
                for (pred_label, label) in zip(pred_labels_np, labels_np):
                    miou, _ = calculate_mIoU(pred_label, label, class_num)
                    batch_miou_sum += miou
                miou_sum += (batch_miou_sum/pred_labels.shape[0])
        val_loss = val_loss_sum/len(val_loader)
        y_val_loss[epoch] = val_loss
        miou_avg = miou_sum/len(val_loader)
        y_avg_miou[epoch] = miou_avg
        print(f"Epoch:{epoch+1}/{epochs} Training loss: {training_loss_avg:.4f} Validation loss: {val_loss:.4f} Validation average mIou: {miou_avg:.4f}")

        if is_StepLR is True:
            scheduler.step()

        if is_early_stopping is True:
            if miou_avg < best_avg_miou:
                best_avg_miou = miou_avg
                no_improvement_epochs = 0
                torch.save(model, f"{dataset_folder_path}/best_model.pth")
            else:
                no_improvement_epochs += 1
            if no_improvement_epochs >= early_stopping_patience:
                print("Early stopping triggered")
                y_training_loss = y_training_loss[0:epoch]
                y_val_loss = y_val_loss[0:epoch]
                y_avg_miou = y_avg_miou[0:epoch]
                raise KeyboardInterrupt

except KeyboardInterrupt as e:
    print(f"Catch exception {e}. Cleaning up...")
finally:
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    del model, criterion, optimizer
    del train_loader, val_loader, test_loader
    gc.collect()
    print("Cleanup complete. Exiting...")

fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(y_training_loss, color='blue', label='Traning loss')
plt.plot(y_val_loss, color='orange', label='Validation loss')
plt.title("Loss")
plt.legend(loc='best')
plt.subplot(1, 2, 2)
plt.plot(y_avg_miou, color='green', label='Average mIou')
plt.legend(loc='best')
plt.title("Average mIou")
plt.show()


test_loader = DataLoader(test_data, batch_size, shuffle=False)
model = torch.load(f"{dataset_folder_path}/best_model.pth")
model.eval()
ious_sum = [0.0]*class_num
miou_sum = 0.0
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc=message, leave=False):
        images = images.to(device)
        labels = labels.to(device).long()

        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        for (pred_label, label) in zip(preds_np, labels_np):
            miou, ious = calculate_mIoU(pred_label, label, class_num)
            miou_sum += miou
            ious_sum=np.add(ious_sum,ious).tolist()
avg_miou=miou_sum/test_data.__len__()
avg_ious=(np.array(ious_sum)/test_data.__len__()).tolist()
print(f"Average mIou of {test_data.__len__()} test images:{avg_miou}")
print(f"Average Iou of {test_data.__len__()} test images:{avg_ious}")
