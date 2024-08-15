# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:20:54 2024

@author: Sen
"""
import numpy as np
import cv2
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

palette=[(0, 0, 0),(60, 180, 75),(255, 225, 25),(0, 130, 200),
         (145, 30, 180),(70, 240, 240),(240, 50, 230),(210, 245, 60),
         (230, 25, 75),(0, 128, 128),(170, 110, 40),(170, 255, 195),
         (128, 128, 0),(250, 190, 190),(0, 0, 128),(128, 128, 128)]

def load_image_and_label(image_path,label_path,target_size):
    image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
    image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size,interpolation=cv2.INTER_LINEAR)
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
    iou = intersection/union if union > 0 else 1
    return iou


def calculate_mIoU(pred_label, label, class_num):
    ious = []
    for index in range(class_num):
        iou = calculate_IoU(pred_label, label, index)
        ious.append(iou)
    return np.mean(np.array(ious)[1:]), ious[1:]

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_folder_path="C:/Users/JARVIS/Documents/Projects/model"
model_name="BiSeNet_best_model.pt"
loaded_model = torch.jit.load(f"{model_folder_path}/{model_name}")
loaded_model.to(device)
loaded_model.eval()

image_path="C:/Users/JARVIS/Documents/Projects/CustomWildScenes2d/test/image/1624325302-559387845.png"
label_path=image_path.replace("image", "newIndexLabel")
target_size=(512,512)
class_num=16

with torch.no_grad():
    image,label=load_image_and_label(image_path,label_path,target_size)
    image_tensor=transforms.ToTensor()(image).unsqueeze(0)
    output = loaded_model(image_tensor.to(device))
    prob = torch.softmax(output, dim=1)
    pred = torch.argmax(prob, dim=1)
    pred_np = pred.cpu().numpy().astype(np.uint8)
    mIoU,ious=calculate_mIoU(pred_np[0],label,class_num)
    print(f"mIoU:{mIoU}")
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

from pathlib import Path
from tqdm import tqdm
import concurrent.futures
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

def update_iou(preds, labels, total_intersection,total_union,class_num):
    for pred, label in zip(preds, labels):
        for index in range(class_num):
            pred_inds = (pred == index)
            label_inds = (label == index)
            intersection = (pred_inds & label_inds).sum().item()
            union = (pred_inds | label_inds).sum().item()
            total_intersection[index] += intersection
            total_union[index] += union
    return total_intersection,total_union

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
model_folder_path="C:/Users/JARVIS/Documents/Projects/model"
model_name="BiSeNet_best_model.pt"
loaded_model = torch.jit.load(f"{model_folder_path}/{model_name}")
loaded_model.to(device)
loaded_model.eval()

data_set_path="C:/Users/JARVIS/Documents/Projects/Custom2WildScenes2d"
loading_mode='pre'
test_data = CustomDataset(data_set_path, 'test',loading_mode)

class_num=16
batch_size=4

test_loader = DataLoader(test_data, batch_size, shuffle=False,drop_last=True)
mIoU=0.0
total_intersection = np.array([0]*class_num,dtype=float)
total_union = np.array([0]*class_num,dtype=float)
with torch.no_grad():
    message = "Test"
    for images, labels in tqdm(test_loader, desc=message, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        outputs = loaded_model(images)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)
        preds_np = preds.cpu().numpy()
        labels_np = labels.cpu().numpy()
        total_intersection,total_union=update_iou(preds_np,labels_np,total_intersection,total_union,class_num)
iou=total_intersection[1:]/(total_union+1e-6)[1:]
mIoU=iou.mean().item()
print(f"mIoU of test set:{mIoU}")
print(f"IoU of test set:{iou}")


