# coding=utf-8
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers import Input, Conv2D,Dense
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.optimizers import Adam
import matplotlib.dates as mdate
import matplotlib.pyplot as plt
import os
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)
# load pima indians dataset
trainx = 'trainx.txt'
trainy = 'trainy.txt'
x_train = np.loadtxt(trainx)
y_train = np.loadtxt(trainy)
x = np.array(x_train)
y = np.array(y_train)
x_train = x.reshape(-1, 2, 2, 3)
y_train = y.reshape(-1, 1, 1, 3)
print(len(x_train))
print(len(y_train))

# =====================================================
testx = 'testx.txt'
testy = 'testy.txt'
x_test = np.loadtxt(testx)
y_test = np.loadtxt(testy)
x = np.array(x_test)
y = np.array(y_test)
x_test = x.reshape(-1, 2, 2, 3)
y_test = y.reshape(-1, 1, 1, 3)
print(x_test.shape)

lr = 0.00005
fname_param = os.path.join('16-4-best.h5')

inputs = Input(shape=(2, 2, 3))
x1 = Conv2D(20, kernel_size=(2,2), input_shape=(None, 2, 2, 3), padding="same", activation='relu')(inputs)
x2 = Conv2D(20, kernel_size=(2,2), padding="same", activation='relu')(x1)
m2 = MaxPooling2D(pool_size=(2,2), padding="same", strides=None)(x2)
x10 = Conv2D(3, kernel_size=(2,2), padding="same", activation='relu')(m2)
# ---------------------------------------------------------
# ---------------------------------------------------------
model = Model(inputs=inputs, outputs=x10)

# Compile model
adam = Adam(lr=lr)
# model.compile(loss='mean_absolute_error', optimizer=adam, metrics=['mean_absolute_error'])
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])
#
print(model.summary())

# early_stopping = EarlyStopping(monitor='mean_absolute_error', patience=7, mode='min')
early_stopping = EarlyStopping(monitor='mean_squared_error', patience=15, mode='min')
# model_checkpoint = ModelCheckpoint(
#                 fname_param, monitor='mean_absolute_error', verbose=2, save_best_only=True, mode='min')
model_checkpoint = ModelCheckpoint(
                fname_param, monitor='mean_squared_error', verbose=2, save_best_only=True, mode='min')
# #
# Fit the model
# model.fit(x_train, y_train, epochs=1000, batch_size=64, verbose=2,callbacks=[early_stopping, model_checkpoint],)
# model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2,callbacks=[ model_checkpoint])
# evaluate the model

# model.save_weights(fname_param, overwrite=True)
# f= os.path.join('22000_41_3_mse_act_22400_3farm.h5')
# model.save_weights(f, overwrite=True)
#
model.load_weights(fname_param)
scores = model.evaluate(x_test, y_test)
print("%s: %.5f" % (model.metrics_names[1], scores[1]))
#calculate predictions
pred = model.predict(x_test)
print("prediction:",pred)
pred=pred.reshape(-1, 1, 1, 3)
a,b,sub,sta=[],[],[],[]
s,mae,mse=0,0,0
j,total=2,400
# j：风电场标号，total：测试总数
while s<total:
    a.append(pred[s][0][0][j]*26.01+0.01)
    b.append(y_test[s][0][0][j]*26.01+0.01)
    s+=1
i=0
print("the mse is：", mean_squared_error(a, b))
print("the mae is：", mean_absolute_error(a, b))

# x_ais=[i for i in range(0,total)]
x_ais=np.arange(0,400,1)
time_file = os.getcwd() + '\\time.txt'
f = open("time_file.txt", "r")
time_tmp = f.read()
time_data = time_tmp.split(",")
print(time_data[0:400])
tmp =[]
# tmp = time_data[0:400]
for i in range(0,400):
    if i%100==0:
        tmp.append(time_data[i])
    else:
        tmp.append(" ")
plt.plot(x_ais, b, color='black',label='Actual value',linewidth=1,markerfacecolor='black',markersize=1)
plt.plot(x_ais, a, color='red',label='Forecasting value',linewidth=1,markerfacecolor='blue',markersize=1)
plt.xticks(x_ais, tmp)
plt.ylabel("Wind speed(m/s)",size=15)
plt.xlabel("Time(10-minute)",size=15)
plt.legend(loc=2)  # 让图例生效
plt.savefig("cnn-4-4.pdf")
plt.show()
