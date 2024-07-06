# -*- coding: utf-8 -*-
"""
Created on Sat Jul  6 22:05:59 2024

@author: Sen
"""
from data_generator import DataGenerator

dg=DataGenerator()
dg.set_parameters("C:/Users/JARVIS/Downloads/WildScenes2d", 0.6, 0, ((0.7,0.1,0.2)))
train,val,test=dg.generate_used_data_path()