# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:10:41 2024

@author: Sen
"""

import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

"""
This file is to generate the split dataset which has a uniform class distribution
"""

def init_custom_dataset_folder(dataset_path,custom_folder_name):
    custom_folder_path=Path(f"{dataset_path}/../{custom_folder_name}")
    if custom_folder_path.exists() is True:
        shutil.rmtree(custom_folder_path.as_posix())
    custom_folder_path.mkdir()
    for item in ['train','val','test']:
        Path(f"{custom_folder_path.as_posix()}/{item}").mkdir()
        Path(f"{custom_folder_path.as_posix()}/{item}/image").mkdir()
        Path(f"{custom_folder_path.as_posix()}/{item}/newIndexLabel").mkdir()

def get_image_path_list(dataset_path,csv_file_name,intrval):
    image_path_list=np.genfromtxt(f"{dataset_path}/{csv_file_name}.csv",dtype=str,delimiter=",",skip_header=1,usecols=(1))
    sampled_image_path_list=[]
    for start in range(0,len(image_path_list),intrval):
        end=min(start+intrval,len(image_path_list))
        index=random.randint(start, end-1)
        sampled_image_path_list.append(image_path_list[index])
    return sampled_image_path_list

def copy_to_custom_dataset_folder(dataset_path,custom_folder_name,sampled_image_path_list,folder_name,target_size=None):
    for path in sampled_image_path_list:
        src_image_path=Path(f"{dataset_path}/../{path}")
        image_name=src_image_path.name
        dest_image_path=f"{dataset_path}/../{custom_folder_name}/{folder_name}/image/{image_name}"
        #shutil.copy(src_image_path.as_posix(), dest_image_path)
        image=cv2.imread(src_image_path.as_posix(),cv2.IMREAD_UNCHANGED)
        src_label_path=Path(src_image_path.as_posix().replace("image", "newIndexLabel"))
        label_name=src_label_path.name
        dest_label_path=f"{dataset_path}/../{custom_folder_name}/{folder_name}/newIndexLabel/{label_name}"
        #shutil.copy(src_label_path.as_posix(), dest_label_path)
        label=cv2.imread(src_label_path.as_posix(),cv2.IMREAD_UNCHANGED)
        if target_size is not None:
            image = cv2.resize(image, target_size,interpolation=cv2.INTER_LINEAR)
            label = cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(dest_image_path, image)
        cv2.imwrite(dest_label_path, label)

def calculate_class_distribution(dataset_path,custom_folder_name,folder_name,class_num):
    new_index_label_folder_path=Path(f"{dataset_path}/../{custom_folder_name}/{folder_name}/newIndexLabel")
    new_label_list=[]
    for item in new_index_label_folder_path.iterdir():
        new_label=cv2.imread(item.as_posix(),cv2.IMREAD_UNCHANGED)
        new_label_list.append(new_label)
    all_pixel_class_num=[]
    for label in new_label_list:
        each_pixel_class_num=[]
        for i in range(class_num):
            each_pixel_class_num.append(np.sum(label == i))
        each_pixel_class_num=np.array(each_pixel_class_num)
        all_pixel_class_num.append(each_pixel_class_num)
    stack=np.vstack(all_pixel_class_num)
    distribution=np.sum(stack,axis=0)
    return distribution
    
dataset_path=f"C:/Users/JARVIS/Documents/Projects/WildScenes2d"
custom_folder_name="CustomWildScenes2d"

init_custom_dataset_folder(dataset_path,custom_folder_name)
random.seed(0)
interval=3
sampled_train_image_path_list=get_image_path_list(dataset_path,'train',interval)
sampled_val_image_path_list=get_image_path_list(dataset_path,'val',interval)
sampled_test_image_path_list=get_image_path_list(dataset_path,'test',interval)

target_size=(512,512)
copy_to_custom_dataset_folder(dataset_path,custom_folder_name,sampled_train_image_path_list,'train',target_size)
copy_to_custom_dataset_folder(dataset_path,custom_folder_name,sampled_val_image_path_list,'val',target_size)
copy_to_custom_dataset_folder(dataset_path,custom_folder_name,sampled_test_image_path_list,'test',target_size)

#Saves paths of smapled images to where you refer to
np.save(f"{dataset_path}/../{custom_folder_name}/sampled_train_image_path_list.npy",sampled_train_image_path_list)
np.save(f"{dataset_path}/../{custom_folder_name}/sampled_val_image_path_list.npy",sampled_val_image_path_list)
np.save(f"{dataset_path}/../{custom_folder_name}/sampled_test_image_path_list.npy",sampled_test_image_path_list)

#If you want to check distributions of sampled images, set this variable to True
is_show=False
if is_show is True:
    class_num=16
    train_distribution=calculate_class_distribution(dataset_path,custom_folder_name,'train',class_num)
    val_distribution=calculate_class_distribution(dataset_path,custom_folder_name,'val',class_num)
    test_distribution=calculate_class_distribution(dataset_path,custom_folder_name,'test',class_num)
    x=np.arange(0,class_num,1)
    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1,3,1)
    plt.plot(x, train_distribution, color='blue', label='train')
    plt.legend(loc='best')
    plt.subplot(1,3,2)
    plt.plot(x, val_distribution, color='blue', label='val')
    plt.legend(loc='best')
    plt.subplot(1,3,3)
    plt.plot(x, test_distribution, color='blue', label='test')
    plt.legend(loc='best')
    plt.title(f"The current sampling is based on the interval of {interval}")
    plt.show()


    
    
    
