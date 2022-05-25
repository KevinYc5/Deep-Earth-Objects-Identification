
import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
import numpy as np
from sklearn.preprocessing import OneHotEncoder
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

#导入训练数据和测试数据
currentpath = os.getcwd()
path = currentpath + '/data'
X_train = np.load(path+"/train_patch.npy")
Y_train = np.load(path+"/train_label.npy")

X_train = np.reshape(X_train, [-1,5,5,1])
Y_train = np.reshape(Y_train,[-1,1])
enc = OneHotEncoder()
enc.fit(Y_train)
Y_train = enc.transform(Y_train).toarray()


num_classes = 2
model = Sequential()

#卷积层1
model.add(Conv2D(64,(3,3),strides=(1,1), padding = 'same',activation='relu', input_shape=(5,5,1),name = "conv1"))
#池化层1 
model.add(MaxPooling2D(pool_size=(2,2)))
#卷积层2
model.add(Conv2D(64,(2,2),padding = 'same',activation='relu', name = "conv2"))
#池化层2
model.add(MaxPooling2D(pool_size=(2,2)))
#Flatten用来将输入“压平”，即把多维的输入一维化，常用在从卷积层到全连接层的过渡
model.add(Flatten())
#Dense为全连接层,前面的参数为神经元个数，后面的为激活函数
model.add(Dense(384, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(192, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#Fit the mode   verbose为日志显示
model.fit(X_train,Y_train,epochs = 20,batch_size= 64,verbose=2,shuffle = True, class_weight='auto')
#模型的存储与读取
model.save(currentpath+'/model/CNN_Model.h5')
























