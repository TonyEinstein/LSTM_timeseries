# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :                                  
# @software   : PyCharm      
# @file       :   window_feature.py
# @Time       :   2021/8/26 17:08


import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
from sklearn.preprocessing import MinMaxScaler,StandardScaler
import datetime
import warnings
warnings.filterwarnings("ignore")#不显示警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
np.set_printoptions(precision=2)


def sequence_engineering_for_train(df_train,arg):
    """
    :param df_train: Dataframe
    :param standard_sign:
    :param windows_size:
    :return:
    """
    df_train.set_index('time',inplace=True)
    array_train_old = df_train.values
    ans = array_train_old[:, 1:]
    scaler = StandardScaler()
    array_train = (array_train_old[:,:1])
    if arg.standard_sign == True:
        scaler.fit(array_train_old[:,:1])
        array_train = scaler.transform(array_train_old[:,:1])
    array_train = np.hstack((array_train, ans))
    features_set = []
    labels = []
    time_list = []
    df_train.reset_index(drop=False, inplace=True)
    # 定义滑动窗口
    for i in range(arg.windows_size,len(df_train),arg.interval+1):
        if i+arg.interval >= len(df_train):
            break
        features_set.append(array_train[i - arg.windows_size:i, :])
        labels.append(array_train_old[i+arg.interval, 0])
        time_list.append(df_train["time"][i+arg.interval])
    features_set, labels = np.array(features_set), np.array(labels)
    labels = np.reshape(labels,[labels.shape[0],1])
    print("sequence train data is prepared")
    return features_set,labels,scaler,time_list

def sequence_engineering_for_test(train_df,test_df,arg,scaler):
    """
    对测试集构造 特征数据，包含了训练集中的最后一个 windows_size 的数据。
    :param train_df: 训练集，Dataframe
    :param test_df: 测试集，Dataframe
    :param windows_size: 滑动窗口大小，需要与训练集的滑动窗口大小一致。
    :return:
    """
    train_df.set_index('time', inplace=True)
    test_df.set_index('time', inplace=True)
    # 获取训练集的最后一个窗口
    test_new_array_old = pd.concat((train_df[-arg.windows_size:], test_df), axis=0).values
    ans = test_new_array_old[:, 1:]
    test_new_array = test_new_array_old[:, :1]
    if arg.standard_sign == True:
        scaler = scaler.fit(test_new_array_old[:,:1])
        test_new_array = scaler.transform(test_new_array_old[:, :1])
    test_new_array = np.hstack((test_new_array, ans))
    test_features = []
    test_labels = []
    time_list = []
    test_df.reset_index(drop=False,inplace=True)
    for i in range(arg.windows_size, len(test_df),arg.interval+1):
        if i+arg.interval >= len(test_df):
            break
        test_features.append(test_new_array[i - arg.windows_size:i, :])
        test_labels.append([test_new_array_old[i+arg.interval, 0]])
        time_list.append(test_df["time"][i + arg.interval])
    test_features = np.array(test_features)
    test_labels = np.array(test_labels)
    print("sequence test data is prepared")
    return test_features,test_labels,scaler,time_list





