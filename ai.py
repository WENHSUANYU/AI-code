
from keras import layers  # 從 keras 套件匯入 layers, models 套件
from keras import models
import pandas as pd
import numpy as np
INPUT_ACTIONS = 5

df = pd.read_csv("Data Set update1.csv",encoding="utf-8")
# x11 = np.array(x1)
model = models.Sequential()
# print(df.head(20))


		     #過濾器數量 ↓      ↓過濾器長寬
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(383
 , 400, 1))) # 加入 Covn2d 層
model.add(layers.MaxPooling2D((2, 2))) # 進行 MaxPooling
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.summary()

#step 2

model.add(layers.Flatten())  # 將 3D 張量展開攤平為 1D, 其 shape = (576, )
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

#step 3
# from keras.datasets import mnist
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
Dependent = df.loc[:, "Class"]    # your "y"d
Independent = df.loc[:, "AttributeX":"AttributeT"] # the rest of the columns (commonly refered to as "X"
X_train, x_test, y_train, y_test = train_test_split(Independent, Dependent,test_size=0.6)
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# df['split'] = np.random.randn(df.shape[0], 1)

# msk = np.random.rand(len(df)) <= 0.6

# train = df[msk]
# test = df[~msk]
X_train = np.array(X_train)
X_train = X_train.reshape(1,383, 8, 1)
X_train = X_train.astype('float32') /255

y_train = np.array(y_train)
y_train = y_train.reshape(383, 2, 1)
y_train = y_train.astype('float32') /255

x_test = to_categorical(x_test)
y_test = to_categorical(y_test)

model.compile(optimizer='rmsprop',
loss='categorical_crossentropy',
metrics=['accuracy'])
model.fit(X_train, x_test, epochs=5, batch_size=64)