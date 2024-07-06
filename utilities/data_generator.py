# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 12:14:14 2024

@author: Sen
"""

from pathlib import Path
import random

class DataGenerator:
    """
    This class is designed for sampling data from given dataset in a uniform distributed way
    """
    def __init__(self):
        self.root_dir=None
        self.dataset_rate=None
        self.random_seed_number=None
        self.split_rate=None
    
    def set_parameters(self,root_dir:str,dataset_rate:float,random_seed_number:int,split_rate:tuple):
        """
        Parameters
        ----------
        root_dir : str
            The root path of dataset.
        dataset_rate : float
            What proportion of dataset you want to be used data.
        random_seed_number : int
            Random seed is used to ensure that the generated data are same every time.
        split_rate : tuple
            To split training data, validation data and test data. Example: (0.7,0.1,0.2) -> (training data, validation data, test data)
        """
        if Path(root_dir).exists() is False:
            raise Exception("Directory is not existing, please check your input")
        self.root_dir=root_dir
        if dataset_rate<=0 and dataset_rate >1:
            raise Exception("Used rate of dataset must be that 0 < dataset_rate <= 1")
        self.dataset_rate=dataset_rate
        self.random_seed_number=random_seed_number
        if len(split_rate)!=3 or sum(split_rate)!=1 or not (split_rate[0]>0 and split_rate[1]>0 and split_rate[2]>0):
            raise Exception("Wrong split parameter")
        self.split_rate=split_rate
        
    def generate_used_data_path(self)->tuple:
        """
        File path structure must be as following 
        
        WildScenes2d
        ├── V-01
        │   ├── image
        │   └── indexLabel
        ├── ...
        │   └── ...
        └── ...
            └── ...
        
        Returns
        -------
        tuple
            Return generated data which include uniform distributed data path with image and index label.

        """
        random.seed(self.random_seed_number)
        sub_folders_dir=self.__search_sub_folders__(self.root_dir)
        training_data_path=[]
        validation_data_path=[]
        test_data_path=[]
        for folder_dir in sub_folders_dir:
            new_folder_dir=f"{folder_dir}/image"
            if Path(new_folder_dir).exists() is True:
                files_path=self.__search_files__(new_folder_dir)
                files_number=len(files_path)
                used_files_number=int(files_number*self.dataset_rate+0.5)
                random_files_numbers = [random.randint(0, files_number-1) for _ in range(used_files_number)]
                used_files_path=[]
                for i in random_files_numbers:
                    used_files_path.append(files_path[i])
                pos0=0
                pos1=int(len(used_files_path)*self.split_rate[0]+0.5)
                training_data_path.extend(used_files_path[pos0:pos1])
                pos2=pos1+int(len(used_files_path)*self.split_rate[1]+0.5)
                validation_data_path.extend(used_files_path[pos1:pos2])
                test_data_path.extend(used_files_path[pos2:])
        training_data_with_index_label_path=[]
        validation_data_with_index_label_path=[]
        test_data_with_index_label_path=[]
        for i in training_data_path:
            new_i=i.replace("/image/", "/indexLabel/")
            training_data_with_index_label_path.append((i,new_i))
        for i in validation_data_path:
            new_i=i.replace("/image/", "/indexLabel/")
            validation_data_with_index_label_path.append((i,new_i))
        for i in test_data_path:
            new_i=i.replace("/image/", "/indexLabel/")
            test_data_with_index_label_path.append((i,new_i))
        return training_data_with_index_label_path,validation_data_with_index_label_path,test_data_with_index_label_path
        
        
    def __search_files__(self,parent_dir:str)->[]:
        directory=Path(parent_dir)
        files_path=[]
        for item in directory.iterdir():
           if item.is_file() is True:
               files_path.append(item.as_posix())
        return sorted(files_path)
        
    def __search_sub_folders__(self,parent_dir:str)->[]:
        directory=Path(parent_dir)
        sub_folders_dir=[]
        for item in directory.iterdir():
            if item.is_dir() is True:
                sub_folders_dir.append(item.as_posix())
        return sub_folders_dir
        
        