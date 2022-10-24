# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :                               
# @software   : PyCharm      
# @file       :   visualization.py
# @Time       :   2021/8/26 17:12

"""
用于数据可视化
"""
import json
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import warnings
import pyecharts.options as opts
from pyecharts.charts import Line
warnings.filterwarnings("ignore")#不显示警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
np.set_printoptions(precision=2)


# 可视化损失值
def chart_loss(x,loss,val_loss,now_time,arg):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="epoch"),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="损失数值",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True,),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='损失值变化')
    )
    line.add_xaxis(xaxis_data=x)
    line.add_yaxis(
        series_name="{0}_loss".format("train"),
        y_axis=loss,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        series_name="{0}_loss".format("val"),
        y_axis=val_loss,
        label_opts=opts.LabelOpts(is_show=False),
    )
    path = os.path.join(arg.evaluate_jpg, arg.them, now_time+"-ws" + str(arg.windows_size) + "_" + arg.mask_choose + "_" + arg.model_sign)
    if not os.path.exists(path):
        os.makedirs(path)
        # os.mkdir(path)
    line.render(os.path.join(path, 'Offset_loss_{0}_interval{1}_{2}_{3}_ws{4}.html'.format(arg.them, arg.interval, arg.model_sign,arg.mask_choose, arg.windows_size)))
    print("chart action is completed!")

# 可视化训练数据
def chart_train(times_list,true,predict,now_time,arg):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="价格",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True,),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='训练集上的预测效果')
    )
    line.add_xaxis(xaxis_data=times_list)
    line.add_yaxis(
        series_name="True Price",
        y_axis=true,
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        series_name="Predicted Price",
        y_axis=predict,
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    path = os.path.join(arg.evaluate_jpg, arg.them,now_time+"-ws" + str(arg.windows_size) + "_" + arg.mask_choose + "_" + arg.model_sign)
    if not os.path.exists(path):
        os.makedirs(path)
    line.render(os.path.join(path,'Offset_train_{0}_interval{1}_{2}_{3}_ws{4}.html'.format(arg.them,arg.interval,arg.model_sign,arg.mask_choose,arg.windows_size)))
    print("train_size chart action is completed!")

# 可视化测试集
def chart_test(time_list,true,predict,now_time,arg):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="价格",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='测试集上的预测效果')
    )
    line.add_xaxis(xaxis_data=time_list)
    line.add_yaxis(
        series_name="True Price",
        y_axis=true.tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.add_yaxis(
        series_name="Predicted Price",
        y_axis=predict.tolist(),
        symbol="emptyCircle",
        is_symbol_show=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
    path = os.path.join(arg.evaluate_jpg, arg.them, now_time+"-ws" + str(arg.windows_size) + "_" + arg.mask_choose + "_" + arg.model_sign)
    if not os.path.exists(path):
        os.makedirs(path)
        # os.mkdir(path)
    line.render(os.path.join(path, 'Offset_test_{0}_interval{1}_{2}_{3}_ws{4}.html'.format(arg.them, arg.interval, arg.model_sign,arg.mask_choose, arg.windows_size)))
    print("test_size chart action is completed!")

# 可视化预测未来的结果
def chart_predict(df,now_time,arg):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="时间"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="价格",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='测试集上的预测效果')
    )
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')
    line.add_xaxis(xaxis_data=[x for x in df['time']])
    line.add_yaxis(
        series_name="True Price",
        y_axis=df.predict.values.tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    path = os.path.join(arg.evaluate_jpg, arg.them, now_time+"-ws" + str(arg.windows_size) + "_" + arg.mask_choose + "_" + arg.model_sign)
    if not os.path.exists(path):
        os.makedirs(path)
        # os.mkdir(path)
    line.render(os.path.join(path, 'Offset_predict_{0}_interval{1}_{2}_{3}_ws{4}.html'.format(arg.them, arg.interval, arg.model_sign,arg.mask_choose, arg.windows_size)))
    print("predict chart action is completed!")

# 可视化差分数据1
def chart_diff(df,name):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="时间"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="价格",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='测试集上的预测效果')
    )
    df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')
    line.add_xaxis(xaxis_data=[x for x in df['time']])
    line.add_yaxis(
        series_name="True Price",
        y_axis=df.price.values.tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.render("diff_{0}.html".format(name))
    print("diff {0} chart action is completed!".format(name))

# 可视化差分数据2
def chart_diff2(df,name):
    # 查看真实值 与预测值的差距：为训练集和测试集准备的函数
    line = Line()
    line.set_global_opts(
        tooltip_opts=opts.TooltipOpts(is_show=True),
        xaxis_opts=opts.AxisOpts(type_="category",name="时间"),
        # xaxis_opts=opts.AxisOpts(type_="category",name="时间",axislabel_opts=opts.LabelOpts(is_show=True,rotate=10)),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            name="价格",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
        ),
        datazoom_opts=opts.DataZoomOpts(is_show=True),
        # datazoom_opts=opts.DataZoomOpts(is_show=True,type_="inside"),
        title_opts=opts.TitleOpts(title='测试集上的预测效果')
    )
    # df['time'] = pd.to_datetime(df['time'], errors='coerce').dt.strftime('%Y-%m-%d')
    line.add_xaxis(xaxis_data=list(range(len(df))))
    line.add_yaxis(
        series_name="True Price",
        y_axis=df.tolist(),
        label_opts=opts.LabelOpts(is_show=False),
    )
    line.render("diff_{0}.html".format(name))
    print("diff {0} chart action is completed!".format(name))



















