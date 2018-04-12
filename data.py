#coding=utf-8
import pandas as pd
import numpy as np
import os

def data_read():
    pwd = os.getcwd()
    print(pwd)
    pwd+='\\data\\train'
    files = os.listdir(pwd)
    dirlist = []
    for fi in files:
        if fi.endswith('.csv'):
            dirlist.append(fi)
    speedall= []
    print(dirlist)
    # dirlist=['C04.csv', 'C05.csv', 'C06.csv', 'C07.csv', 'C08.csv']
    # for fi in dirlist:

    dir0 = pwd+'\\'+dirlist[0]
    dir1 = pwd+'\\'+dirlist[1]
    # dir2 = pwd+'\\'+dirlist[2]
    dir3 = pwd+'\\'+dirlist[3]
    # print(dir0,dir1,dir2,dir3)
    data0 = pd.read_csv(dir0,encoding='gbk')
    data1 = pd.read_csv(dir1,encoding='gbk')
    # data2 = pd.read_csv(dir2,encoding='gbk')
    data3 = pd.read_csv(dir3,encoding='gbk')

    data_old0 = data0.values
    data_old1 = data1.values
    # data_old2 = data2.values
    data_old3 = data3.values
    # lenmin=min(len(data_old0),len(data_old1),len(data_old2),len(data_old3))
    lenmin=min(len(data_old0),len(data_old1),len(data_old3))
    speedtmp0, speedtmp1,speedtmp3=[],[],[]
    tmp = []
    for i in range(0,lenmin):
        if data_old0[i,2]>0:
            if data_old1[i,2]>0:
                    if data_old3[i,2]>0:
                        speedtmp0.append(data_old0[i,2])
                        speedtmp1.append(data_old1[i,2])
                        # speedtmp2.append(data_old2[i,2])
                        speedtmp3.append(data_old3[i,2])
                        tmp.append(data_old3[i,0])
    print("len of 0:",len(speedtmp0))
    print("len of 1:",len(speedtmp1))
    # print("len of 2:",len(speedtmp2))
    print("len of 3:",len(speedtmp3))

    speedall.append(speedtmp0)
    speedall.append(speedtmp1)
    # speedall.append(speedtmp2)
    speedall.append(speedtmp3)
    print(len(speedall))

    rol=4
    i,ma,mi=0,0,0
    # while i<len(speedall[i]):
    while i<1:
        a=max(speedall[i])
        b=min(speedall[i])
        if a>ma:
            ma=a
        if b>mi:
            mi=b
        i+=1
    print(ma)
    print(mi)
    i=0
    while i<len(speedall):
        j=0
        while j<len(speedall[i]):
            speedall[i][j] = (speedall[i][j] - mi) / (ma - mi)
            j+=1
        i+=1
    i=0
# # ------------------------------------------------------------------
# #     读取数据的不同仅仅只是这一部分分注释
    # print(minlen)
    j=0
    # //train
    xtrain,ytrain,xtest,ytest=[],[],[],[]
    while j < 20000:
        i=0
        while i<len(speedall):
            xtrain.append(speedall[i][j:j+rol])
            ytrain.append(speedall[i][j+rol+3:j+rol+4])
            i+=1
        j+=1
    x_train=np.array(xtrain)
    y_train=np.array(ytrain)
    # print(x_train[0])
    np.savetxt("trainx.txt",x_train)
    np.savetxt("trainy.txt",y_train)
    print(x_train.shape)
    print(y_train.shape)
# # //////////////////////////////////////////////////////test data
    j=22000
    # print(tmp[22000])
    while j < 22400:
        i=0
        while i<len(speedall):
            xtest.append(speedall[i][j:j+rol])
            ytest.append(speedall[i][j+rol+3:j+rol+4])
            i+=1
        j+=1
    x_test=np.array(xtest)
    y_test=np.array(ytest)
    # print(x_train[0])
    np.savetxt("testx.txt",x_test)
    np.savetxt("testy.txt",y_test)
    # print(tmp[22000:22400])
    time_file = os.getcwd()+'\\time.txt'
    f=open("time_file.txt","w")
    time_str = ""
    for i in tmp[22000:22400]:
        i=i[5:]
        time_str+=i+","
    f.write(time_str)
    print(time_str)
    print(x_test.shape)

data_read()