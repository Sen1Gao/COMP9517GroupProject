# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:10:41 2024

@author: Sen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

"""
This file is to generate the split dataset which has a uniform class distribution
"""

#Your own csv files folder path
csv_file_folder_path="../WildScenes-main/data/splits/opt2d"

train_image_list=np.genfromtxt(f"{csv_file_folder_path}/train.csv",dtype=str,delimiter=",",skip_header=1,usecols=(1))
val_image_list=np.genfromtxt(f"{csv_file_folder_path}/val.csv",dtype=str,delimiter=",",skip_header=1,usecols=(1))
test_image_list=np.genfromtxt(f"{csv_file_folder_path}/test.csv",dtype=str,delimiter=",",skip_header=1,usecols=(1))

#Uses this variable to sample image with interval referred by yourself
intrval=9
split_train_image_list=train_image_list[::intrval]
split_val_image_list=val_image_list[::intrval]
split_test_image_list=test_image_list[::intrval]

#If you want to check distributions of split images, set this variable to True
distribution_display=False
if distribution_display is True:
    combined_split_image_list=[split_train_image_list,split_val_image_list,split_test_image_list]
    class_distribution=[]
    for image_list in combined_split_image_list:
        new_label_list=[]
        for image_path in image_list:
            new_label_path=f"C:/Users/JARVIS/Documents/Projects/{image_path.replace('image','newIndexLabel')}"
            new_label=cv2.imread(new_label_path,cv2.IMREAD_UNCHANGED)
            new_label_list.append(new_label)
        distribution = np.sum(np.vstack([np.sum(a == c) for c in range(16)] for a in new_label_list), axis=0)
        class_distribution.append(distribution)
    x=np.arange(0,16,1)
    fig = plt.figure(figsize=(10, 10))
    plt.subplot(2,2,1)
    plt.plot(x, class_distribution[0], color='blue', label='train')
    plt.legend(loc='best')
    plt.subplot(2,2,2)
    plt.plot(x, class_distribution[1], color='blue', label='val')
    plt.legend(loc='best')
    plt.subplot(2,2,3)
    plt.plot(x, class_distribution[2], color='blue', label='test')
    plt.legend(loc='best')
    plt.show()

#Saves paths of split images to where you refer to
saving_path=f"C:/Users/JARVIS/Documents/Projects/WildScenes2d"
np.save(f"{saving_path}/split_train_image_list.npy",split_train_image_list)
np.save(f"{saving_path}/split_val_image_list.npy",split_val_image_list)
np.save(f"{saving_path}/split_test_image_list.npy",split_test_image_list)
