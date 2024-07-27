# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 19:50:54 2024

@author: Sen
"""
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate
from tensorflow.keras.models import Model
import matplotlib.pyplot as plt

#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

root_dir="C:/Users/JARVIS/Documents/Projects"
split_train_image_list=np.load(f"{root_dir}/WildScenes2d/split_train_image_list.npy")
split_val_image_list=np.load(f"{root_dir}/WildScenes2d/split_val_image_list.npy")
split_test_image_list=np.load(f"{root_dir}/WildScenes2d/split_test_image_list.npy")

def data_loading(image_paths,batch_size,target_size):
    while True:
        for start in range(0,len(image_paths),batch_size):
            end=min(start+batch_size,len(image_paths))
            batch_images=[]
            batch_labels=[]
            for i in range(start,end):
                image_path=f"{root_dir}/{image_paths[i]}"
                image=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
                image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image=cv2.resize(image, target_size)
                image=image.astype(np.float32)/255.0
                batch_images.append(image)
                label_path=f"{root_dir}/{image_paths[i]}".replace("image", "newIndexLabel")
                label=cv2.imread(label_path,cv2.IMREAD_UNCHANGED)
                label=cv2.resize(label, target_size,interpolation=cv2.INTER_NEAREST)
                #label=label.astype(np.int32)
                label=tf.one_hot(label,depth=16)
                batch_labels.append(label)
            yield np.array(batch_images),np.array(batch_labels)

def unet_model(input_size,class_num):
    inputs=Input(shape=input_size)
    
    #encode
    c1=Conv2D(32,(3,3),activation='relu',padding='same')(inputs)
    #c1=Conv2D(64,(3,3),activation='relu',padding='same')(c1)
    p1=MaxPooling2D((2,2))(c1)
    
    c2=Conv2D(64,(3,3),activation='relu',padding='same')(p1)
    #c2=Conv2D(128,(3,3),activation='relu',padding='same')(c2)
    p2=MaxPooling2D((2,2))(c2)
    
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(p2)
    #c3 = Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)
    
    c4 = Conv2D(256, (3, 3), activation='relu', padding='same')(p3)
    #c4 = Conv2D(512, (3, 3), activation='relu', padding='same')(c4)
    p4 = MaxPooling2D((2, 2))(c4)
    
    c5 = Conv2D(512, (3, 3), activation='relu', padding='same')(p4)
    #c5 = Conv2D(1024, (3, 3), activation='relu', padding='same')(c5)
    
    # decode
    u6 = UpSampling2D((2, 2))(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(256, (3, 3), activation='relu', padding='same')(u6)
    #c6 = Conv2D(512, (3, 3), activation='relu', padding='same')(c6)
    
    u7 = UpSampling2D((2, 2))(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(128, (3, 3), activation='relu', padding='same')(u7)
    #c7 = Conv2D(256, (3, 3), activation='relu', padding='same')(c7)
    
    u8 = UpSampling2D((2, 2))(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(64, (3, 3), activation='relu', padding='same')(u8)
    #c8 = Conv2D(128, (3, 3), activation='relu', padding='same')(c8)
    
    u9 = UpSampling2D((2, 2))(c8)
    u9 = concatenate([u9, c1])
    c9 = Conv2D(32, (3, 3), activation='relu', padding='same')(u9)
    #c9 = Conv2D(64, (3, 3), activation='relu', padding='same')(c9)
    
    outputs = Conv2D(class_num, (1, 1), activation='softmax')(c9)
    
    model = Model(inputs, outputs)
    return model

input_size=(512,512,3)
class_num=16
model=unet_model(input_size,class_num)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

batch_size=4
target_size=(512,512)
train_data_generator=data_loading(split_train_image_list,batch_size,target_size)
val_data_generator=data_loading(split_val_image_list,batch_size,target_size)
history=model.fit(train_data_generator,
          steps_per_epoch=len(split_train_image_list)//batch_size,
          validation_data=val_data_generator,
          validation_steps=len(split_val_image_list)//batch_size,
          epochs=50,
          verbose=1)


fig = plt.figure(figsize=(15, 10))  
plt.subplot(1,2,1)
plt.plot(history.epoch, history.history['loss'], color='blue', label='Traning loss')
plt.plot(history.epoch, history.history['val_loss'], color='orange', label='Validation loss')
plt.ylim((0,1))
plt.legend(loc='best')
plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.epoch, history.history['accuracy'], color='blue', label='Traning accuracy')
plt.plot(history.epoch, history.history['val_accuracy'], color='orange', label='Validation accuracy')
plt.ylim((0,1))
plt.legend(loc='best')
plt.title("Accuracy")
plt.show()
            
tmp=split_test_image_list[15]
image_path=f"{root_dir}/{tmp}"
img=cv2.imread(image_path,cv2.IMREAD_UNCHANGED)
img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img=cv2.resize(img, target_size)
img=img.astype(np.float32)/255.0
imgs=np.reshape(img, (1,512,512,3))
predictions = model.predict(imgs)        
pred=predictions[0]

rst=np.zeros((pred.shape[0],pred.shape[1]))
for i in range(pred.shape[0]):
    for j in range(pred.shape[1]):
        arr=pred[i][j]
        index=np.argmax(arr)
        rst[i,j]=index
rst=cv2.resize(rst,(2016,1512) ,interpolation=cv2.INTER_NEAREST)
cv2.imwrite("C:/Users/JARVIS/Documents/Projects/WildScenes2d/rst.png", rst)


