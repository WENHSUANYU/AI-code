#!/usr/bin/env python
# coding: utf-8

# 初始化一個小型卷積神經網路

# In[1]:


from keras.layers import Convolution1D, MaxPooling1D

from keras import layers  # 從 keras 套件匯入 layers, models 套件
from keras import models
import pandas as pd
import numpy as np


df = pd.read_csv("Data.csv",encoding="utf-8")
model = models.Sequential()
		     #過濾器數量 ↓      ↓過濾器長寬
model.add(layers.Conv1D(256, kernel_size=1, strides=1,activation='relu', input_shape=(1
 , 4))) # 加入 Covn1d 層
model.add(layers.MaxPooling1D(1, strides=1)) # 進行 MaxPooling
model.add(layers.Conv1D(128, (1), activation='relu'))
model.add(layers.MaxPooling1D(1, strides=1))
model.add(layers.Conv1D(64, (1), activation='relu'))
model.summary()


# 在卷積神經網路上加入分類器

# In[2]:


model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(4,activation='softmax'))
model.summary()


#  MNIST 影像訓練卷積神經網路

# In[3]:


from sklearn.model_selection import train_test_split 

df = np.array(df)
np.random.shuffle(df)
Dependent = df[:, 4:8] #y
Independent = df[:, 0:4] # the rest of the columns (commonly refered to as "X"

X_train, x_test, y_train, y_test = train_test_split(Independent, Dependent,test_size=0.4,random_state=0)

print(X_train.shape) 
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape)
print("--------------------------------------")
X_train = X_train[:,np.newaxis]
X_train = X_train.astype('float32')/100

x_test = x_test[:,np.newaxis] 
x_test = x_test.astype('float32')/100

y_train = y_train[:,np.newaxis]
y_train = y_train.astype('float32')/100

y_test = y_test[:,np.newaxis]
y_test = y_test.astype('float32')/100

print(X_train.shape) 
print(x_test.shape) 
print(y_train.shape) 
print(y_test.shape)

model.compile(optimizer='Adam',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, y_train,validation_data=(x_test,y_test) ,epochs=5, batch_size=4)

