# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 23:30:23 2024

@author: Sen
"""

from pathlib import Path
import numpy as np
import cv2

"""
This file is designed to verify whether a specifical index of color is in an indexLabel by searching
"""
# the range of color_index is from 1 to 18
color_index=1

directory=Path("C:/Users/JARVIS/Downloads/WildScenes2d")
sub_folders_dir=[]
for item in directory.iterdir():
    if item.is_dir() is True:
        sub_folders_dir.append(f"{item.as_posix()}/indexLabel")

files_path=[]    
for sub_folder_dir in sub_folders_dir:
    directory=Path(sub_folder_dir)
    for item in directory.iterdir():
       if item.is_file() is True:
           files_path.append(item.as_posix())

for file_path in files_path:
    tmp=cv2.imread(file_path,cv2.IMREAD_UNCHANGED)
    if color_index in tmp:
        print(file_path)