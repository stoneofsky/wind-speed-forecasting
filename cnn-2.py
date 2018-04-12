# coding=utf-8
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.models import Model
from keras.layers import Input, Conv2D,Dense
from keras.layers.core import Activation
from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
import math

plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
# fix random seed for reproducibility
rol=4
# load pima indians dataset
trainx = 'trainx.txt'
trainy = 'trainy.txt'
x_train = np.loadtxt(trainx)
y_train = np.loadtxt(trainy)
x = np.array(x_train)
y = np.array(y_train)
x_train = x.reshape(-1, rol, rol, 3)
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
x_test = x.reshape(-1, rol, rol, 3)
y_test = y.reshape(-1, 1, 1, 3)
print(x_test.shape)

lr = 0.0005
fname_param = os.path.join('16-1-best.h5')

inputs = Input(shape=(rol, rol, 3))
x1 = Conv2D(20, kernel_size=(2,2), input_shape=(None, rol, rol, 3), padding="valid", activation='relu')(inputs)
x2 = Conv2D(20, kernel_size=(2,2), padding="valid", activation='relu')(x1)
x10 = Conv2D(3, kernel_size=(2,2), padding="valid", activation='relu')(x2)

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
model.fit(x_train, y_train, epochs=500, batch_size=64, verbose=2,callbacks=[ model_checkpoint])
# evaluate the model

model.save_weights(fname_param, overwrite=True)
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
    a.append(pred[s][0][0][j]*20.61)
    b.append(y_test[s][0][0][j]*20.61)
    s+=1
i=0
print("the mse is：", mean_squared_error(a, b))
print("the mae is：", mean_absolute_error(a, b))
x_ais=[i for i in range(0,total)]
plt.plot(x_ais, b, color='black',label='test',linewidth=1,markerfacecolor='black',markersize=1)
plt.plot(x_ais, a, color='red',label='pred',linewidth=1,markerfacecolor='blue',markersize=1)
plt.ylabel("CNN数据对比")
plt.legend()  # 让图例生效
#
plt.savefig("cnn16-3.pdf")
plt.show()
