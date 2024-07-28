# -*- coding: utf-8 -*-
"""
Created on Wed Jul 24 23:10:41 2024

@author: Sen
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
import shutil

"""
This file is to generate the split dataset which has a uniform class distribution
"""

dataset_dir=f"C:/Users/JARVIS/Documents/Projects"
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
            new_label_path=f"{dataset_dir}/{image_path.replace('image','newIndexLabel')}"
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

new_dataset_path=f"{dataset_dir}/CustomWildScenes2d"
new_dataset_dir=Path(new_dataset_path)
if new_dataset_dir.exists() is True:
    shutil.rmtree(new_dataset_dir.as_posix())
new_dataset_dir.mkdir()
Path(f"{new_dataset_path}/train").mkdir()
Path(f"{new_dataset_path}/train/image").mkdir()
Path(f"{new_dataset_path}/train/newIndexLabel").mkdir()
Path(f"{new_dataset_path}/val").mkdir()
Path(f"{new_dataset_path}/val/image").mkdir()
Path(f"{new_dataset_path}/val/newIndexLabel").mkdir()
Path(f"{new_dataset_path}/test").mkdir()
Path(f"{new_dataset_path}/test/image").mkdir()
Path(f"{new_dataset_path}/test/newIndexLabel").mkdir()


#Saves paths of split images to where you refer to
np.save(f"{new_dataset_path}/split_train_image_list.npy",split_train_image_list)
np.save(f"{new_dataset_path}/split_val_image_list.npy",split_val_image_list)
np.save(f"{new_dataset_path}/split_test_image_list.npy",split_test_image_list)

for i in split_train_image_list:
    src_image_path=Path(f"{dataset_dir}/{i}")
    image_name=src_image_path.name
    dest_image_path=f"{new_dataset_path}/train/image/{image_name}"
    shutil.copy(src_image_path.as_posix(), dest_image_path)
    src_label_path=Path(src_image_path.as_posix().replace("image", "newIndexLabel"))
    label_name=src_label_path.name
    dest_label_path=f"{new_dataset_path}/train/newIndexLabel/{label_name}"
    shutil.copy(src_label_path.as_posix(), dest_label_path)

for i in split_val_image_list:
    src_image_path=Path(f"{dataset_dir}/{i}")
    image_name=src_image_path.name
    dest_image_path=f"{new_dataset_path}/val/image/{image_name}"
    shutil.copy(src_image_path.as_posix(), dest_image_path)
    src_label_path=Path(src_image_path.as_posix().replace("image", "newIndexLabel"))
    label_name=src_label_path.name
    dest_label_path=f"{new_dataset_path}/val/newIndexLabel/{label_name}"
    shutil.copy(src_label_path.as_posix(), dest_label_path)
    
for i in split_test_image_list:
    src_image_path=Path(f"{dataset_dir}/{i}")
    image_name=src_image_path.name
    dest_image_path=f"{new_dataset_path}/test/image/{image_name}"
    shutil.copy(src_image_path.as_posix(), dest_image_path)
    src_label_path=Path(src_image_path.as_posix().replace("image", "newIndexLabel"))
    label_name=src_label_path.name
    dest_label_path=f"{new_dataset_path}/test/newIndexLabel/{label_name}"
    shutil.copy(src_label_path.as_posix(), dest_label_path)
    
    
    
    
    
