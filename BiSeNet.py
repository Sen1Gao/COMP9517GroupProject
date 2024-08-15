# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:50:54 2024

@author: Sen
"""

from pathlib import Path
from tqdm import tqdm
import concurrent.futures
from datetime import datetime

import numpy as np
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset


class CustomDataset(Dataset):
    def __init__(self, dataset_folder_path, csv_file_type, loading_mode, target_size=None):
        self.dataset_folder_path = dataset_folder_path
        self.csv_file_type = csv_file_type
        self.sampled_image_path_list = np.load(f"{dataset_folder_path}/sampled_{csv_file_type}_image_path_list.npy")
        self.loading_mode = loading_mode
        self.target_size = target_size
        self.transform = transforms.ToTensor()
        self.image_list = [None]*self.__len__()
        self.label_list = [None]*self.__len__()
        if self.loading_mode == 'pre':
            self.__pre_load_and_process__()

    def __process_image__(self, image,label):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.target_size is not None:
            image = cv2.resize(image, self.target_size,interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, self.target_size,interpolation=cv2.INTER_NEAREST)
        return image,label

    def __load_image__(self, image_name):
        image_path = f"{self.dataset_folder_path}/{self.csv_file_type}/image/{image_name}"
        image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        label_name = image_name
        label_path = f"{self.dataset_folder_path}/{self.csv_file_type}/newIndexLabel/{image_name}"
        label = cv2.imread(label_path, cv2.IMREAD_UNCHANGED)
        return self.__process_image__(image,label)

    def __load_image_and_index__(self,args):
        index,image_name=args
        image,label=self.__load_image__(image_name)
        return index,image,label

    def __pre_load_and_process__(self):
        print(f"Start loading {self.csv_file_type} data")
        args_list=[(index,Path(image_path).name) for index,image_path in enumerate(self.sampled_image_path_list)]
        results=None
        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = list(executor.map(self.__load_image_and_index__, args_list))
        for index,image,label in results:
            self.image_list[index]=image
            self.label_list[index]=label
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
            image, label = self.__load_image__(image_name)
        tensor_image = self.transform(image)
        tensor_label = torch.from_numpy(label).long()
        return tensor_image, tensor_label


class Trainer():
    def __init__(self,classifier,class_num,is_printing_model=False):
        self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Current used device is {self.device}")
        self.class_num=class_num
        self.model=classifier(class_num)
        if is_printing_model is True:
            print("Model Information:")
            print(self.model)
        self.device_model=self.model.to(self.device)
        self.train_data=None
        self.val_data=None
        self.test_data=None
        self.learning_rate=None
        self.batch_size=None
        self.epochs=None
        self.ignore_index=None
        
    def __update_iou__(self,preds, labels, total_intersection,total_union):
        for pred, label in zip(preds, labels):
            for index in range(self.class_num):
                pred_inds = (pred == index)
                label_inds = (label == index)
                intersection = (pred_inds & label_inds).sum().item()
                union = (pred_inds | label_inds).sum().item()
                total_intersection[index] += intersection
                total_union[index] += union
        return total_intersection,total_union
        
    def load_image_data(self,dataset_folder_path,loading_mode,target_size=None):
        self.train_data = CustomDataset(dataset_folder_path, 'train', loading_mode, target_size)
        self.val_data = CustomDataset(dataset_folder_path, 'val',loading_mode, target_size)
        self.test_data = CustomDataset(dataset_folder_path, 'test',loading_mode, target_size)
    
    def set_training_parameters(self,learning_rate=0.01,batch_size=16,epochs=50,ignore_index=-100):
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.epochs=epochs
        self.ignore_index=ignore_index

    def start_training(self,model_save_path):
        torch.cuda.empty_cache()
        y_training_loss = np.array([0]*self.epochs, dtype=float)
        y_val_loss = np.array([0]*self.epochs, dtype=float)
        y_miou = np.array([0]*self.epochs, dtype=float)
        criterion = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        optimizer = optim.Adam(self.device_model.parameters(), lr=self.learning_rate,weight_decay=1e-5)
        train_loader = DataLoader(self.train_data, self.batch_size, shuffle=False,drop_last=True)
        val_loader = DataLoader(self.val_data, self.batch_size, shuffle=False,drop_last=True)
        best_miou=0
        for epoch in range(self.epochs):
            message = f"Current epoch:{epoch+1}/{self.epochs} Training"
            self.device_model.train()
            batch_training_loss_sum = 0.0
            for images, labels in tqdm(train_loader, desc=message, leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                optimizer.zero_grad()
                outputs = self.device_model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                batch_training_loss_sum += loss.item()
            avg_training_loss = batch_training_loss_sum / len(train_loader)
            y_training_loss[epoch] = avg_training_loss

            message = f"Current epoch:{epoch+1}/{self.epochs} validating"
            self.device_model.eval()
            batch_val_loss_sum = 0.0
            mIoU=0.0
            total_intersection = np.array([0]*self.class_num,dtype=float)
            total_union = np.array([0]*self.class_num,dtype=float)
            with torch.no_grad():
                for images, labels in tqdm(val_loader, desc=message, leave=False):
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.device_model(images)
                    loss = criterion(outputs, labels)
                    batch_val_loss_sum += loss.item()
                    pred_labels = torch.argmax(outputs, dim=1)
                    pred_labels_np = pred_labels.cpu().numpy()
                    labels_np = labels.cpu().numpy()
                    total_intersection,total_union=self.__update_iou__(pred_labels_np,labels_np,total_intersection,total_union)
            iou=total_intersection[1:]/(total_union+1e-6)[1:]
            mIoU=iou.mean().item()
            if mIoU>best_miou:
                best_miou=mIoU
                self.save_current_model(model_save_path,is_final_model=False)
            avg_val_loss = batch_val_loss_sum/len(val_loader)
            y_val_loss[epoch] = avg_val_loss
            y_miou[epoch]=mIoU
            print(f"Epoch:{epoch+1}/{self.epochs} Training loss: {avg_training_loss:.4f} Validation loss: {avg_val_loss:.4f} mIoU of Validation: {mIoU:.4f}")
        self.save_current_model(model_save_path)
        return y_training_loss,y_val_loss,y_miou
    
    def draw_training_result(self,train_loss,val_loss,mIoU):
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_loss, color='blue', label='Traning loss')
        plt.plot(val_loss, color='orange', label='Validation loss')
        plt.title("Loss")
        plt.legend(loc='best')
        plt.subplot(1, 2, 2)
        plt.plot(mIoU, color='green', label='mIou')
        plt.title("Mean Iou")
        plt.legend(loc='best')
        plt.show()
        
    def start_testing(self):
        torch.cuda.empty_cache()
        test_loader = DataLoader(self.test_data, self.batch_size, shuffle=False,drop_last=True)
        self.device_model.eval()
        mIoU=0.0
        total_intersection = np.array([0]*self.class_num,dtype=float)
        total_union = np.array([0]*self.class_num,dtype=float)
        with torch.no_grad():
            message = "Test"
            for images, labels in tqdm(test_loader, desc=message, leave=False):
                images = images.to(self.device)
                labels = labels.to(self.device)
                outputs = self.device_model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                preds_np = preds.cpu().numpy()
                labels_np = labels.cpu().numpy()
                total_intersection,total_union=self.__update_iou__(preds_np,labels_np,total_intersection,total_union)
        iou=total_intersection[1:]/(total_union+1e-6)[1:]
        mIoU=iou.mean().item()
        print(f"mIoU of test set:{mIoU}")

    def save_current_model(self,model_save_path,is_final_model=True):
        scripted_model = torch.jit.script(self.device_model)
        if is_final_model is True:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime('%Y-%m-%d-%H-%M-%S')
            scripted_model.save(f"{model_save_path}/{timestamp_str}_model.pt")
        else:
            scripted_model.save(f"{model_save_path}/best_model.pt")
        

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SpatialPath(nn.Module):
    def __init__(self):
        super(SpatialPath, self).__init__()
        self.conv1 = ConvBlock(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.conv2 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.conv3 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.dropout1=nn.Dropout2d(0.1)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
    def forward(self, x):
        x=self.conv1(x)
        x=self.dropout1(x)
        x=self.conv2(x)
        x=self.dropout2(x)
        x=self.conv3(x)
        x=self.dropout3(x)
        return x


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(AttentionRefinementModule, self).__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(in_channels, out_channels, 1, 1, 0)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg=self.global_pool(x)
        avg=self.conv(avg)
        avg=self.bn(avg)
        attention = self.sigmoid(avg)
        x=self.conv(x)
        x=self.bn(x)
        x=self.relu(x)
        return x * attention


class ContextPath(nn.Module):
    def __init__(self):
        super(ContextPath, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.arm8x = AttentionRefinementModule(128,512)
        self.arm16x = AttentionRefinementModule(256,256)
        self.conv1=ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1)
        self.conv2=ConvBlock(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.dropout1=nn.Dropout2d(0.1)
        self.dropout2=nn.Dropout2d(0.2)
        self.dropout3=nn.Dropout2d(0.3)
        
    def forward(self, x):
        x2 = self.resnet18.conv1(x)
        x2 = self.resnet18.bn1(x2)
        x2 = self.resnet18.relu(x2)
        x4 = self.resnet18.maxpool(x2)
        x4 = self.resnet18.layer1(x4)
        x8 = self.resnet18.layer2(x4)  
        x16 = self.resnet18.layer3(x8)
        
        avg=self.avg_pool(x16)
        avg_up=F.interpolate(avg, size=x16.size()[2:], mode='bilinear', align_corners=True)
        
        x16_arm=self.arm16x(x16)
        x16_arm=x16_arm+avg_up
        x16_arm_up=F.interpolate(x16_arm, size=x8.size()[2:], mode='bilinear', align_corners=True)
        x16_arm_up=self.conv1(x16_arm_up)
        x16_arm_up=self.dropout2(x16_arm_up)
        
        x8_arm=self.arm8x(x8)
        x8_arm=x8_arm+x16_arm_up
        x8_arm=self.conv2(x8_arm)
        x8_arm=self.dropout3(x8_arm)
        
        return x8_arm,x


class FeatureFusionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureFusionModule, self).__init__()
        self.conv = ConvBlock(in_channels, out_channels, 1, 1, 0)
        self.dropout=nn.Dropout2d(0.4)
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels // 4, out_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, sp, cp):
        fusion = torch.cat([sp, cp], dim=1)
        fusion = self.conv(fusion)
        fusion=self.dropout(fusion)
        attention = self.attention(fusion)
        return fusion + fusion * attention


class BiSeNet(nn.Module):
    def __init__(self, num_classes):
        super(BiSeNet, self).__init__()
        self.spatial_path = SpatialPath()
        self.context_path = ContextPath()
        self.feature_fusion = FeatureFusionModule(512,128)
        self.final_conv=nn.Conv2d(128, num_classes, 1,1,0)
        
    def forward(self, x):
        sp = self.spatial_path(x)
        cp,x = self.context_path(x)
        out = self.feature_fusion(sp,cp)
        out = F.interpolate(out, size=x.size()[2:], mode='bilinear', align_corners=True)
        out = self.final_conv(out)
        return out

trainer=Trainer(BiSeNet,16)

data_set_path="C:/Users/JARVIS/Documents/Projects/Custom2WildScenes2d"
trainer.load_image_data(data_set_path, "pre")

trainer.set_training_parameters(0.001,8,50,0)

model_save_path=data_set_path
y1,y2,y3=trainer.start_training(model_save_path)

trainer.draw_training_result(y1, y2, y3)

trainer.start_testing()
