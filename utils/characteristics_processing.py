# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :                                
# @software   : PyCharm      
# @file       :   characteristics_processing.py
# @Time       :   2021/8/27 9:07
import sys

import matplotlib.pyplot as plt
import pandas as pd
import numpy as  np
import datetime
from statsmodels.tsa.stattools import adfuller as ADF
from statsmodels.stats.diagnostic import acorr_ljungbox
from dateutil import rrule
import warnings
warnings.filterwarnings("ignore")#不显示警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
np.set_printoptions(precision=2)

# 差分
def difference_methods(df):
    df[u'price'] = df[u'price'].astype(float)
    D_data = df.diff().dropna()
    return D_data
# 反差分
def reverse_diff(result,labels_last):
    for i in range(len(result)):
        result[i][0] = labels_last + result[i][0]
        labels_last = result[i][0]
    return result

# 数据检验：平稳性、白噪声、差分
def inspection_handling(df,diff_sign):
    # 为了防止LSTM的缺点：滞后性；所以必须判断是否进行差分  或者强制差分
    if diff_sign == True:
        # 如果是True那么就是强制至少进行了一次拆分。
        df = difference_methods(df)
        diff_sign = False
        df = inspection_handling(df,diff_sign)
        print("-------------\t已进行强制差分运算!\t-------------")
    # 平稳性检验-ADF
    adf = ADF(df[u'price'])[1]
    if adf >= 0.05:
        # 差分
        df = difference_methods(df)
        print("-------------\t已进行差分运算!\t-------------")
        inspection_handling(df,diff_sign)
    # 白噪声检验
    al = list( acorr_ljungbox(df, lags=1)[1])[0]
    if al >= 0.05:
        print(u'差分序列的白噪声检验结果为：', acorr_ljungbox(df, lags=1))
        print("该序列为白噪声序列，没有往下建模的必要了！")
        sys.exit()
    return df

# 差分操作调用函数
def diff_operation(df_original,diff_sign):
    df = df_original.set_index('time')
    df_diff = inspection_handling(df,diff_sign)
    df_diff.reset_index(inplace=True)
    print("inspection_handled Dataframe lenght is：", len(df_diff))
    return df_diff

# 计算两个日期之间有几个旬
def count_xun(time1, time2):
    date_range = pd.date_range(time1, time2, freq='D')
    tmp = []
    count_ver = 0
    if len(date_range) <= 0:
        date_range = pd.date_range(time2,time1,freq='D').to_list()
    for i in range(len(date_range)):
        if str(date_range[i]).split(" ")[0][-2:] =='05':
            count_ver += 1
        elif str(date_range[i]).split(" ")[0][-2:] =='15':
            count_ver += 1
        elif str(date_range[i]).split(" ")[0][-2:] =='25':
            # print("执行了")
            count_ver += 1
        else:
            pass
        tmp.append(str(date_range[i]).split(" ")[0])
    return count_ver,tmp

# 判断有多少个周末：
def jude_week(time_frame):
    time_frame.set_index('time', drop=True, inplace=True)
    # 星期几
    time_frame["weekday"] = time_frame.index.weekday
    # 是否是周末
    time_frame['is_weekend'] = time_frame.weekday.isin([5, 6]) * 1
    time_frame.reset_index(drop=False, inplace=True)
    return time_frame

# 添加并计算有多少额外预测的时间
def add_incremental_time(predict_last_time,predict_more_n,time_frame,them,interval):
    predict_last_time = datetime.datetime.strptime(predict_last_time, "%Y-%m-%d")
    if them == '日':
        predict_last_time_end = predict_last_time + datetime.timedelta(days=predict_more_n)
        more_time_frame = pd.date_range(predict_last_time, predict_last_time_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        more_time_frame = fit_time2(more_time_frame, them)
        time_frame = pd.concat([time_frame,more_time_frame],axis=0)
        time_frame.reset_index(drop=True,inplace=True)
        return time_frame
    elif them == "周":
        predict_last_time = predict_last_time + datetime.timedelta(days=7)
        predict_last_time_end =  predict_last_time + datetime.timedelta(days=predict_more_n*5+interval*5)
        more_time_frame = pd.date_range(predict_last_time, predict_last_time_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        more_time_frame = fit_time2(more_time_frame, them)
        time_frame = pd.concat([time_frame, more_time_frame], axis=0)
        time_frame.reset_index(drop=True, inplace=True)
        return time_frame
    elif them == "旬":
        # predict_last_time = predict_last_time + datetime.timedelta(days=15)
        predict_last_time_end = predict_last_time + datetime.timedelta(days=predict_more_n * 10+interval*15)
        more_time_frame = pd.date_range(predict_last_time, predict_last_time_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        more_time_frame = fit_time2(more_time_frame, them)
        time_frame = pd.concat([time_frame, more_time_frame], axis=0)
        time_frame.reset_index(drop=True, inplace=True)
        return time_frame
    elif them == "胖":
        predict_last_time_end = predict_last_time + datetime.timedelta(days=predict_more_n * 15+interval*15)
        more_time_frame = pd.date_range(predict_last_time, predict_last_time_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        more_time_frame = fit_time2(more_time_frame, them)
        time_frame = pd.concat([time_frame, more_time_frame], axis=0)
        time_frame.reset_index(drop=True, inplace=True)
        return time_frame
    elif them == "月":
        predict_last_time_end = predict_last_time + datetime.timedelta(days=predict_more_n * 30+interval*30)
        more_time_frame = pd.date_range(predict_last_time, predict_last_time_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        more_time_frame = fit_time2(more_time_frame, them)
        time_frame = pd.concat([time_frame, more_time_frame], axis=0)
        time_frame.reset_index(drop=True, inplace=True)
        return time_frame
    else:
        pass
    return time_frame


# 计算两个时间  的 相差 数值，（若是天、计算相差天数；若是旬，计算相差旬数）
def time_diff(time1, time2,time_theme,diff_sign,predict_more_n):
    """
    :param time1: type:str
    :param time2: type：str
    :return:
    """
    offset = predict_more_n
    if diff_sign == True:
        # 有差分的情况
        offset = 0
    time_array1 = datetime.datetime.strptime(time1, "%Y-%m-%d")
    time_array2 = datetime.datetime.strptime(time2, "%Y-%m-%d")
    if time_theme == '日':
        # 2013-12-22~2013-12-25的天数为 23 24 25，也就是不包括22
        result = abs((time_array1 - time_array2).days)+offset
        print("两个日期的间隔天数（包含偏移量）：{} ".format(result))
        return result
    elif time_theme == '周':
        result = abs(int(datetime.datetime.strftime(time_array1, "%W")) - int(datetime.datetime.strftime(time_array2, "%W")))+offset
        print("两个日期的间隔周数（包含偏移量）：{} ".format(result))
        return result
    elif time_theme == '旬':
        time1 = datetime.datetime.strptime(time1, "%Y-%m-%d")
        time1 = time1 + datetime.timedelta(days=10)
        time1 = time1.strftime("%Y-%m-%d")
        count_num,time_list = count_xun(time1,time2)
        # print(count_num)
        count_num = count_num + offset
        print("两个日期的间隔旬数（包含偏移量）：{} ".format(count_num))
        return count_num
    elif time_theme == '月':
        month_num = rrule.rrule(rrule.MONTHLY, dtstart=time_array1, until=time_array2).count()
        if time1[:4] == time2[:4] :
            month_num -= 1
        month_num += offset
        print("两个日期的间隔月数（包含偏移量）：{} ".format(month_num))
        return month_num
    elif time_theme =='胖':
        time_frame = pd.date_range(time1, time2, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        time_frame = fit_time2(time_frame, time_theme)
        half_month = 0
        half_month = half_month + offset + len(time_frame)
        print("两个日期的间隔半月数（包含偏移量）：{} ".format(half_month))
        return half_month
    # 其它：季度、年 的时间主题 不加入计算也不加入建模，因为数据太少。直接跳过。
    else:
        pass

def fit_time(data_last_time,predict_last_time,time_theme):
    time_frame = pd.date_range(data_last_time, predict_last_time, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
    time_frame['values'] = 3
    if time_theme == '日':
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '周':
        data_week = time_frame.resample('W', on="time").mean()
        data_week.reset_index(inplace=True)
        data_week = data_week.drop(columns=['values'])
        return data_week
    elif time_theme == '旬':
        data_last_time = data_last_time + datetime.timedelta(days=6)
        time_frame = pd.date_range(data_last_time, predict_last_time, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        time_frame['values'] = 3
        time_frame["年"] = time_frame["time"].dt.year
        time_frame["月"] = time_frame["time"].dt.month
        time_frame["日"] = time_frame["time"].dt.day
        time_frame["flag"] = time_frame["日"]
        time_frame["xun"] = np.where((time_frame["flag"] > 10) & (time_frame["flag"] <= 20), "中旬", np.where(time_frame["flag"] <= 10, "上旬", "下旬"))
        time_frame['年'] = time_frame['年'].astype(str)
        time_frame['月'] = time_frame['月'].astype(str)
        time_frame['旬'] = time_frame['年'] + '-' + time_frame['月'] + '-' + time_frame['xun']
        time_frame = time_frame[['旬', 'values']]
        for i in range(len(time_frame)):
            if time_frame['旬'][i][-2:] == '上旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '05'
            elif time_frame['旬'][i][-2:] == '中旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '15'
            elif time_frame['旬'][i][-2:] == '下旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '25'
            else:
                print("发生错误")
                pass
        time_frame['time'] = pd.to_datetime(time_frame['旬'])
        time_frame = time_frame.groupby('time').mean('values')
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '月':
        time_frame = time_frame.resample('M', on="time").mean()
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '胖':
        # print(type(data_last_time))
        # sys.exit()
        data_last_time = datetime.datetime.strptime(data_last_time, "%Y-%m-%d")
        data_last_time = data_last_time + datetime.timedelta(days=10)
        time_frame = pd.date_range(data_last_time, predict_last_time, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
        time_frame['values'] = 3
        time_frame['time'] = pd.to_datetime(time_frame['time']).dt.strftime('%Y-%m-%d')
        for i in range(len(time_frame)):
            if float(time_frame['time'][i][-2:]) >= 16.0:
                time_frame['time'][i] = str(time_frame['time'][i][:-2]) + "20"
                continue
            time_frame['time'][i] = str(time_frame['time'][i][:-2]) + "10"
        time_frame = time_frame.groupby('time').mean()
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    # 其它：季度、年 的时间主题 不加入计算也不加入建模，因为数据太少。直接跳过。
    else:
        return None


def fit_time2(time_frame,time_theme):
    time_frame['values'] = 3
    if time_theme == '日':
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '周':
        data_week = time_frame.resample('W', on="time").mean()
        data_week.reset_index(inplace=True)
        data_week = data_week.drop(columns=['values'])
        return data_week
    elif time_theme == '旬':
        time_frame["年"] = time_frame["time"].dt.year
        time_frame["月"] = time_frame["time"].dt.month
        time_frame["日"] = time_frame["time"].dt.day
        time_frame["flag"] = time_frame["日"]
        time_frame["xun"] = np.where((time_frame["flag"] > 10) & (time_frame["flag"] <= 20), "中旬", np.where(time_frame["flag"] <= 10, "上旬", "下旬"))
        time_frame['年'] = time_frame['年'].astype(str)
        time_frame['月'] = time_frame['月'].astype(str)
        time_frame['旬'] = time_frame['年'] + '-' + time_frame['月'] + '-' + time_frame['xun']
        time_frame = time_frame[['旬', 'values']]
        for i in range(len(time_frame)):
            if time_frame['旬'][i][-2:] == '上旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '05'
            elif time_frame['旬'][i][-2:] == '中旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '15'
            elif time_frame['旬'][i][-2:] == '下旬':
                time_frame['旬'][i] = time_frame['旬'][i][:-2] + '25'
            else:
                print("发生错误")
                pass
        time_frame['time'] = pd.to_datetime(time_frame['旬'])
        time_frame = time_frame.groupby('time').mean('values')
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '月':
        time_frame = time_frame.resample('M', on="time").mean()
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    elif time_theme == '胖':
        time_frame['time'] = pd.to_datetime(time_frame['time']).dt.strftime('%Y-%m-%d')
        for i in range(len(time_frame)):
            if float(time_frame['time'][i][-2:]) >= 16.0:
                time_frame['time'][i] = str(time_frame['time'][i][:-2]) + "20"
                continue
            time_frame['time'][i] = str(time_frame['time'][i][:-2]) + "10"
        time_frame = time_frame.groupby('time').mean()
        time_frame.reset_index(inplace=True)
        time_frame = time_frame.drop(columns=['values'])
        return time_frame
    # 其它：季度、年 的时间主题 不加入计算也不加入建模，因为数据太少。直接跳过。
    else:
        return None

