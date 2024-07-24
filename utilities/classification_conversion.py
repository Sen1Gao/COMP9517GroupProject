# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 20:50:51 2024

@author: Sen
"""

from pathlib import Path
import numpy as np
import cv2

"""
This file is to convert classes from 1+18 to 1+15, where 1 indicates the unlabeled class
"""

root_folder_path=Path("C:/Users/JARVIS/Documents/Projects/WildScenes2d")
for folder_path in root_folder_path.iterdir():
    new_index_label_folder_path=f"{folder_path.as_posix()}/newIndexLabel"
    Path(new_index_label_folder_path).mkdir()
    index_label_folder_path=f"{folder_path.as_posix()}/indexLabel"
    for label_path in Path(index_label_folder_path).iterdir():
        label=cv2.imread(label_path.as_posix(),cv2.IMREAD_UNCHANGED)
        label[label==1]=6
        label[label==12]=16
        label[label==13]=0
        
        #label[label==0]=0
        label[label==2]=1
        label[label==3]=2
        label[label==4]=3
        label[label==5]=4
        label[label==6]=5
        label[label==7]=6
        label[label==8]=7
        label[label==9]=8
        label[label==10]=9
        label[label==11]=10
        label[label==14]=11
        label[label==15]=12
        label[label==16]=13
        label[label==17]=14
        label[label==18]=15
        cv2.imwrite(f"{new_index_label_folder_path}/{label_path.name}", label)
        

