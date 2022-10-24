# !/usr/bin python3                                 
# encoding    : utf-8 -*-                            
# @author     :                                 
# @software   : PyCharm      
# @file       :   IPO_prediction.py
# @Time       :   2021/8/26 17:05
import os.path
import random
import sys

import matplotlib.pyplot as plt
import numpy as  np
import tensorflow as tf
from adtk.data import validate_series
from adtk.visualization import plot
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout
import argparse
from tensorflow.keras.callbacks import ModelCheckpoint
import warnings
from adtk.detector import InterQuartileRangeAD


tf.keras.backend.set_epsilon(1)
warnings.filterwarnings("ignore")#不显示警告信息
plt.rcParams['font.sans-serif'] = ['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False #用来正常显示负号
from sklearn.metrics import explained_variance_score,r2_score
np.set_printoptions(precision=2)
from utils.visualization import *
from utils.window_feature import *
from utils.characteristics_processing import *

now_time = datetime.datetime.now().strftime('%YY-%mM-%dD %HH:%Mm:%Ss').replace(" ","_").replace(":","_")
print("tensorflow version is {}".format(tf.__version__))
print("numpy version is {}".format(np.__version__))
print("pandas version is {}".format(pd.__version__))

def setup_seed(seed):
    random.seed(seed)  # 为python设置随机种子
    np.random.seed(seed)  # 为numpy设置随机种子
    tf.random.set_seed(seed)  # tf cpu fix seed
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # tf gpu fix seed, please `pip install tensorflow-determinism` first


setup_seed(1234)


# 计算测试集预测的mape
def mape(y_true, y_pred):
    return np.mean(np.abs((y_pred - y_true) / y_true)) * 100

def predict_more_way(test_df,model,last_test_feature,scaler,last_prediction,predictions,arg):
    # 准备多预测n个的那些时间
    if arg.standard_sign==True:
        last_prediction = scaler.transform(np.array(predictions[-1]).reshape([1,1]))
    last_test_time = test_df['time'].iloc[-1]
    last_test_time = pd.to_datetime(last_test_time)
    if arg.them == "周":
        test_time_opration = last_test_time + datetime.timedelta(days=0)
        print("最后计算的起始时间：", test_time_opration)
        day_end = test_time_opration + datetime.timedelta(days=arg.predict_more_n * 7)
    elif arg.them == "旬":
        test_time_opration = last_test_time + datetime.timedelta(days=6)
        print("最后计算的起始时间：", test_time_opration)
        day_end = test_time_opration + datetime.timedelta(days=arg.predict_more_n * 5)
    elif arg.them == "胖":
        test_time_opration = last_test_time + datetime.timedelta(days=10)
        print("最后计算的起始时间：", test_time_opration)
        day_end = test_time_opration + datetime.timedelta(days=arg.predict_more_n * 10)
    elif arg.them == "月":
        test_time_opration = last_test_time + datetime.timedelta(days=1)
        print("最后计算的起始时间：", test_time_opration)
        day_end = test_time_opration + datetime.timedelta(days=arg.predict_more_n * 30)
    else:
        test_time_opration = last_test_time + datetime.timedelta(days=1)
        print("最后计算的起始时间：", test_time_opration)
        day_end = test_time_opration + datetime.timedelta(days=arg.predict_more_n * 90)
    time_frame = pd.date_range(test_time_opration, day_end, freq='D', name='time',closed='right').to_frame().reset_index(drop=True)
    if arg.predict_more_n != 0:
        time_frame = fit_time2(time_frame, arg.them)
    last_test_feature = np.delete(last_test_feature, 0, axis=0)
    last_test_feature = np.append(last_test_feature, last_prediction, axis=0)
    # 预测 测试集之外的第一个
    predict_value = model.predict(np.expand_dims(last_test_feature, axis=0))
    predictions = np.append(predictions, predict_value, axis=0)
    # 预测测试集之外的除去第0个的从第1个开始~第n个的预测
    if arg.predict_more_n != 0:
        for i in range(1, arg.predict_more_n):
            # 删除掉上一个特征
            last_test_feature = np.delete(last_test_feature, 0, axis=0)
            # print(last_test_feature)
            last_prediction = np.array(predictions[-1]).reshape([1, 1])
            if arg.standard_sign == True:
                last_prediction = scaler.transform(np.array(predictions[-1]).reshape([1, 1]))
            # print(last_prediction)
            # 向窗口加入下一个特征
            last_test_feature = np.append(last_test_feature, last_prediction, axis=0)
            # print(last_test_feature)
            # 预测
            predict_value = model.predict(np.expand_dims(last_test_feature, axis=0))
            # 拼接到预测数组
            predictions = np.append(predictions, predict_value, axis=0)
    else:
        pass
        # predictions = np.delete(predictions, -1)
    predictions = predictions[arg.predict_more_n:]
    return predictions

def evaluate_test(model,test_df,test_features,test_labls,train_features, train_labels,arg,scaler,test_time,train_time):
    # 预测测试集
    # global test_time_opration
    predictions = model.predict(test_features)
    last_test_feature = test_features[-1]#测试集的最后一个
    last_prediction = np.array(predictions[-1]).reshape([1, 1])
    # 提前预测策略
    if arg.predict_advance_n > 0:
        # 提前预测的部分:predict_advance_n
        features_advance_n = train_features[-arg.predict_advance_n:]
        prediction_advance_n = model.predict(features_advance_n)
        # 提前预测实现逻辑
        predictions = np.vstack((prediction_advance_n, predictions))
        predictions = np.delete(predictions, list(range(len(predictions))[-arg.predict_advance_n:]))

    # 进行延迟预测：
    predictions = predict_more_way(test_df, model, last_test_feature, scaler, last_prediction, predictions, arg)

    # 拼接
    result_df = pd.DataFrame(columns=["time","true_price","predict_price"])
    # print(test_df['time'].tolist())
    result_df["time"] = test_time
    result_df["true_price"] = test_labls.reshape(1,-1)[0].tolist()
    result_df["predict_price"] = predictions.reshape(1,-1)[0].tolist()
    # 测试集上的模型评估: 第一个是Loss，第二个是mape
    evaluate_values_list = model.evaluate(test_features,test_labls)
    r2 = r2_score(test_labls,predictions)
    evs = explained_variance_score(test_labls, predictions)
    chart_test(test_time, test_labls, predictions, now_time,arg)
    result_df["true_price"] = round(result_df["true_price"], 2)
    result_df["predict_price"] = round(result_df["predict_price"], 2)
    if not os.path.exists(os.path.join(arg.evaluate_pd,arg.them)):
        os.makedirs(os.path.join(arg.evaluate_pd,arg.them))
    result_df.to_csv(os.path.join(arg.evaluate_pd,arg.them,"{0}_ws{1}_{2}_{3}_{4}.csv".format(arg.them,arg.windows_size, now_time, arg.model_sign, arg.mask_choose)),encoding='utf-8', index=False)
    print("save csv file is completed!")
    # 返回损失和r2分数、评估指标
    return evaluate_values_list[0],r2,evs

def load_model(model_path,model_name,theme):
    print("loading this model！it's name is {0}".format(os.path.join(model_path,theme,model_name)))
    new_model = tf.keras.models.load_model(os.path.join(model_path,theme,model_name))
    print("load model is completed!")
    return new_model

def predict_feature_values(model,features_set,labels,scaler,data_last_time,arg):
    global now_time
    # n = 2
    # 下面是一个包含时间的dataframe：未来要预测的时间
    # time_frame = pd.date_range(data_last_time.strftime('%Y-%m-%d'), predict_last_time, freq='D',name='time',closed='right').to_frame().reset_index(drop=True)
    # print(time_frame)
    time_frame_feature = fit_time(data_last_time,arg.predict_last_time,arg.them)
    print("要预测的未来时间是：\n",time_frame_feature)
    # 把多预测的时间部分根据时间主题them 加到time_frame里面:返回只有时间列
    time_frame = add_incremental_time(arg.predict_last_time,arg.predict_more_n,time_frame_feature,arg.them,arg.interval)#此函数还需要完善
    time_frame_list = [time_frame['time'][t] for t in range(len(time_frame))]
    if type(time_frame["time"][0]) != str:
        time_frame_list = [time_frame['time'][t].strftime('%Y-%m-%d') for t in range(len(time_frame))]
    print("未来预测的时间段+多预测的n时间：", time_frame_list)
    print("要预测的一共：\t{0}\t天".format(len(time_frame_list)))
    ft = []
    sg = 0
    for t in time_frame_list:
        sg += 1
        if sg % (arg.interval+1)==0:
            ft.append(t)
    print("经过interval运算后的时间：", ft)
    # 制作第一个Input X：
    last_features_set=features_set[-1]
    last_predict = [labels[-1]]
    # 循环预测：
    result = np.empty(shape=(1, 1))
    for i in range(len(ft)):
        if arg.standard_sign == True:
            last_predict = scaler.transform(last_predict)
        last_test_feature = np.delete(last_features_set, 0, axis=0)
        last_test_feature = np.append(last_test_feature, last_predict, axis=0)
        last_predict = model.predict(np.expand_dims(last_test_feature, axis=0))
        #循环拼接predict进行存储
        result = np.append(result,last_predict,axis=0)
    # 把前N个给去掉
    result = np.delete(result, 0, axis=0).tolist()
    if len(result) >1 and len(result)>arg.predict_more_n:
        try:
            del result[:arg.predict_more_n]
            del ft[-arg.predict_more_n:]
        except Exception as e:
            print(e)

    result = np.array(result)
    ft = np.array(ft)
    # 反差分：转换回 标准化之前 的 差分后的数据，这里不需要，因为无需拆分，这里用不到
    if arg.diff_sign == True:
        result = reverse_diff(result,labels[-1])
    # 将真实值和预测值 拼接起来存档
    result = pd.DataFrame(result,columns=["predict"])
    ft = pd.DataFrame(ft,columns=["time"])
    data_concat = pd.concat([ft, result], axis=1)
    # 保存预测文件
    data_concat['predict'] = round(data_concat['predict'], 2)
    print(data_concat)
    chart_predict(data_concat,now_time,arg)
    if not os.path.exists(os.path.join(arg.evaluate_pd,arg.them,"predict")):
        os.makedirs(os.path.join(arg.evaluate_pd,arg.them,"predict"))
    data_concat.to_csv(os.path.join(arg.evaluate_pd,arg.them,'predict',"{0}_{1}_ws{2}_{3}_{4}.csv".format(arg.them,arg.windows_size, now_time, arg.model_sign,arg.mask_choose)), encoding='utf-8',index=False)
    print("predict is completed!")
    print("save csv file is completed!")
    return data_concat

def find_abnormal(df,c):
    df = df.set_index(keys='time')
    df.index = pd.to_datetime(df.index)
    df = validate_series(df)
    iqr_ad = InterQuartileRangeAD(c=1.5)
    anomalies = iqr_ad.fit_detect(df)
    plot(df, anomaly=anomalies, ts_markersize=2, anomaly_markersize=4, anomaly_tag="marker", anomaly_color='red')
    return df

def train_model(time_list, features_set, labels, arg):
    global now_time
    """
    :param features_set: 训练数据集
    :param labels: 训练的标签
    :param batch_size: 每批大小
    :param epochs: 训练轮数
    :return:
    """
    # print("loss:",type(loss),loss)
    model = Sequential()
    #CNN+LSTM
    # model.add(tf.keras.layers.Conv1D(filters=20, kernel_size=4, strides=2, padding="valid",input_shape=(features_set.shape[1], features_set.shape[2])))
    # model.add(LSTM(units=arg.units1, return_sequences=True, kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dropout(arg.dropout))
    # model.add(LSTM(units=arg.units2, kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dense(units=arg.units_last))


    # LSTM
    model.add(LSTM(units=arg.units1, return_sequences=True, input_shape=(features_set.shape[1], features_set.shape[2]),kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    model.add(Dropout(arg.dropout))
    model.add(LSTM(units=arg.units2,kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    model.add(Dense(units=arg.units_last))

    # RNN
    # model.add(SimpleRNN(arg.units1, return_sequences=True, input_shape=(features_set.shape[1], features_set.shape[2]),kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dropout(arg.dropout))
    # model.add(SimpleRNN(units=arg.units2,kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dense(units=arg.units_last))

    #线性回归：
    # model.add(Flatten(input_shape=[features_set.shape[1], features_set.shape[2]]))
    # model.add(Dense(units=arg.units1,kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dropout(arg.dropout))
    # model.add(Dense(units=arg.units2, kernel_regularizer=tf.keras.regularizers.l2(arg.l2)))
    # model.add(Dense(units=arg.units_last))

    model_name = "{0}_{1}_{2}_{3}ep_{4}ws_{5}bs_{6}_{7}.h5".format(arg.them, arg.model_sign, now_time, arg.epochs,
                                                                   arg.windows_size, arg.batch_size, arg.optimizer, arg.mask_choose)
    # 指数衰减学习率
    exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=arg.learning_rate,
                                                                       decay_steps=arg.decay_steps, decay_rate=arg.decay_rate)
    if not os.path.exists(arg.log_path):
        os.makedirs(arg.log_path)
    if not os.path.exists(os.path.join(arg.model_path,arg.them)):
        os.makedirs(os.path.join(arg.model_path,arg.them))
    callbacks = [
        # 当验证集上的损失“val_loss”连续n个训练回合（epoch）都没有变化，则提前结束训练
        tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=arg.min_delta, patience=arg.patience, mode='auto'),
        # 使用TensorBoard保存训练的记录，保存到“./logs”目录中
        tf.keras.callbacks.TensorBoard(log_dir=arg.log_path, histogram_freq=2, write_images=True, update_freq='epoch',
                                       profile_batch=5),
        ModelCheckpoint(filepath=os.path.join(arg.model_path,arg.them,model_name), monitor='val_loss', mode='auto', verbose=1,
                        save_best_only=True)
    ]
    if arg.beta_jude == False:
        model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay), loss=arg.loss, metrics=arg.metrics)
    else:
        model.compile(optimizer=tf.keras.optimizers.Adam(exponential_decay,beta_1=arg.beta_1, beta_2=arg.beta_2), loss=arg.loss, metrics=arg.metrics)

    history = model.fit(features_set, labels, validation_split=arg.val_size, callbacks=callbacks, epochs=arg.epochs, verbose=2,batch_size=arg.batch_size)
    # print(history.history)
    # print(model.history.history.keys())
    # sys.exit()
    model_cofig = model.get_config()
    predict_train = model.predict(features_set)
    print(predict_train.shape)
    print(labels.shape)
    # 计算r方分数
    r2_value = r2_score(labels, predict_train)
    # 计算解释方差
    exs_value = explained_variance_score(labels, predict_train)
    evaluation_dict = dict()
    chart_train(time_list, labels.tolist(), predict_train.tolist(), now_time,arg)
    evaluation_dict['model type'] = arg.model_sign
    evaluation_dict['model name'] = model_name
    evaluation_dict['data them'] = arg.them
    evaluation_dict["train mode"] = arg.mask_choose
    evaluation_dict['train time'] = now_time
    evaluation_dict['windows_size'] = arg.windows_size
    evaluation_dict["epoch"] = arg.epochs
    evaluation_dict["dropout"] = arg.dropout
    evaluation_dict["l1"] = arg.l1
    evaluation_dict["l2"] = arg.l2
    evaluation_dict["epochs"] = arg.epochs
    evaluation_dict["batch size"] = arg.batch_size
    evaluation_dict['optimizer'] = arg.optimizer
    evaluation_dict['r2 score on train'] = r2_value
    evaluation_dict['explained variance score on train'] = exs_value
    evaluation_dict['avg huber loss on train'] = sum(history.history["loss"]) / len(history.history["loss"])
    evaluation_dict['min huber loss on train'] = min(history.history["loss"])
    evaluation_dict['max huber loss on train'] = max(history.history["loss"])
    evaluation_dict["avg huber val_loss on validation"] = sum(history.history['val_loss']) / len(history.history['val_loss'])
    evaluation_dict['min huber val_loss on validation'] = min(history.history['val_loss'])
    evaluation_dict['max huber val_loss on validation'] = max(history.history['val_loss'])
    print("模型名字：\t",model_name)
    print("R²：",r2_value)
    chart_loss(list(range(len(history.history['loss']))), history.history['loss'],  history.history['val_loss'],now_time,arg)
    return model,evaluation_dict,model_cofig

def parameter_initialization():
    parser = argparse.ArgumentParser(description='模型声明以及各个参数、变量描述！正在初始化参数.....')
    parser.add_argument('--mask_choose', type=str, default='predict', help='part 为模型训练部分数据，full 为模型训练所有数据,predict为预测功能,')
    parser.add_argument('--data_file_path', type=str, default="data\data.csv", help="数据文件的路径。")
    parser.add_argument('--holiday_path', type=str, default="holiday", help="节假日的日期存放的文件夹，这个不用理会")
    parser.add_argument('--config_path', type=str, default="config", help="存储模型配置信息、评价指标信息的文件夹")
    parser.add_argument('--output_path', type=str, default="output", help="输出文件夹")
    parser.add_argument('--them', type=str, default="月", help="数据的主题：日 周 旬 胖 月 季")
    parser.add_argument('--load_model_name', type=str,default='月_LSTM_2022Y-10M-25D_01H_20m_48s_300ep_6ws_8bs_adam_full.h5', help="加载的模型名字")
    parser.add_argument('--model_sign', type=str, default='LSTM', help="给模型、文件、jpg命名的时候标明的模型名字")
    parser.add_argument('--test_size', type=int, default=0.1, help="测试集的大小比例")
    parser.add_argument('--val_size', type=float, default=0.1, help="验证集在训练集中的大小比例")
    #
    parser.add_argument('--windows_size', type=int, default=6, help="滑动窗口的大小,【滑动窗口大小 最好比 预测未来的 个数大，这样的话可以用历史数据加入对未来的预测】")
    parser.add_argument('--epochs', type=int, default=300, help="模型最大训练轮数")
    parser.add_argument('--batch_size', type=int, default=8, help="一个训练批次的大小")
    parser.add_argument('--optimizer', type=str, default='adam', help="模型优化器选项")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="模型优化器的 学习率")
    parser.add_argument('--beta_1', type=float, default=0.9, help="模型优化器的 动量参数1")
    parser.add_argument('--beta_2', type=float, default=0.96, help="模型优化器的 动量参数2")
    parser.add_argument('--beta_jude', type=bool, default=True, help="是否需要动量")
    parser.add_argument('--l2', type=float, default=0.01, help="L2正则化项")
    parser.add_argument('--l1', type=float, default=0.1, help="L1正则化项")
    parser.add_argument('--dropout', type=float, default=0.1, help="dropout的比例")
    parser.add_argument('--units1', type=int, default=32, help="单元1内部隐藏层的大小")
    parser.add_argument('--units2', type=int, default=64, help="单元2内部隐藏层的大小")
    parser.add_argument('--units_last', type=int, default=1, help="最后一层的units大小")
    parser.add_argument('--decay_steps', type=int, default=1, help="模型优化器 经历个多少个step后完成一次学习率衰减 ")
    parser.add_argument('--decay_rate', type=float, default=0.96, help="模型优化器的 学习率衰减系数")
    parser.add_argument('--loss', type=str, default=tf.keras.losses.MeanSquaredError(), help="损失函数")
    parser.add_argument('--metrics', type=list, default=['mean_squared_error'], help="评估指标")
    parser.add_argument('--log_path', type=str, default='logs', help="日志路径")
    parser.add_argument('--model_path', type=str, default='model', help="模型存放路径")
    parser.add_argument('--evaluate_jpg', type=str, default='evaluate_jpg', help="存储 evaluate 产生的图片文件")
    parser.add_argument('--evaluate_pd', type=str, default='evaluate_predict_data', help="存储 evaluate 产生的数据文件")
    parser.add_argument('--diff_sign', type=bool, default=False, help="是否进行强制差分运算")
    parser.add_argument('--standard_sign', type=bool, default=True, help="是否进行标准化")
    parser.add_argument('--min_delta', type=float, default=5e-3, help="根据损失函数提前停止网络的损失值变化 最小判断标准")
    parser.add_argument('--patience', type=int, default=10, help="根据损失函数值 提前停止网络的 连续最大相同轮数")
    parser.add_argument('--predict_last_time', type=str, default='2021-05-30', help="预测范围的最后一个时间节点【真实值数据文件的最后一个时间点】")
    # 使用延迟预测的方式
    parser.add_argument('--predict_more_n', type=int, default=1, help="比 predict_last_time 的时间点多预测 n 个单位")
    # 使用提前预测的方法,减少误差累积
    parser.add_argument('--predict_advance_n', type=int, default=0, help="这也是滞后性数值，比 predict_last_time 的时间点提前预测 n 个单位,可在测试集使用,也可在预测未来时使用")
    parser.add_argument('--interval', type=int, default=0, help="每个多少个时间单位预测一次，鸡的生长周期，默认是2，主要用于月度单位")
    args = parser.parse_args()
    print("Parameter initialization already underway!")
    return args

"""
part模式：是训练-验证-测试 进行训练然后评估历史数据的建模情况。

full模式： 是使用所有数据划分训练-验证  进行训练得到要预测未来的模型。

predict模式：使用full模式的模型进行预测未来。

predict_last_time：是full模式里面的整个数据集的最后一个时间点

"""




def enter():
    arg = parameter_initialization()
    df_original = pd.read_csv(arg.data_file_path)
    print("load csv file is ok!")
    print("original Dataframe lenght is :",len(df_original))
    data=df_original.copy()
    c = 1.5
    df = find_abnormal(data,c)
    print(df)

    if not os.path.exists(arg.output_path):
        os.makedirs(arg.output_path)
    run_dir = os.path.join(arg.output_path,"run_"+now_time)
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)
    arg.config_path = os.path.join(run_dir,arg.config_path)
    arg.log_path = os.path.join(run_dir,arg.log_path)
    arg.evaluate_jpg = os.path.join(run_dir,arg.evaluate_jpg)
    arg.evaluate_pd = os.path.join(run_dir,arg.evaluate_pd)
    # 判断是否需要进行差分运算
    # diff_sign = arg.diff_sign
    # data = diff_operation(df_original.copy(), arg.diff_sign)
    # chart_diff(data,"1")


    if not os.path.exists(arg.config_path):
        os.makedirs(arg.config_path)
    #获取训练模式
    if  arg.mask_choose == 'part':
        arg.model_path = os.path.join(run_dir, arg.model_path)
        # 准备训练数据
        test_size = int(len(data) * arg.test_size)
        train, test = data[:-test_size], data[-test_size:]
        arg.decay_steps = len(train) // arg.batch_size
        # 训练集
        train_features, train_labels, scaler_train,train_time = sequence_engineering_for_train(train.copy(),arg)
        # 测试集
        test_features,test_labls, scaler_test,test_time= sequence_engineering_for_test(train.copy(), test.copy(),arg,scaler_train)

        # print(train_features[:10])
        # print(train_labels[:10])
        # sys.exit()
        model,evaluate_dict,model_config = train_model(train_time, train_features, train_labels, arg)
        print("model is fited!")
        print("part：训练-测试 模型训练完毕！")
        print("测试集的时间：",test_time)

        huber_loss,r2,evs= evaluate_test(model,test,test_features,test_labls,train_features, train_labels,arg,scaler_train,test_time,train_time)
        evaluate_dict['huber loss on test'] = huber_loss
        evaluate_dict['r2 score on test'] = r2
        evaluate_dict['"explained variance score on test'] = evs
        print(evaluate_dict)
        # 存储评价指标：
        with open(os.path.join(arg.config_path, "info_part_{0}.json".format(arg.them)), mode='a', encoding='utf-8') as f:
            json.dump(evaluate_dict, f, indent=4, ensure_ascii=False)
        # 存储模型的config
        # with open(os.path.join(arg.config_path, "model_config.json"), mode='a', encoding='utf-8') as f:
        #     json.dump(model_config, f, ensure_ascii=False)
    elif  arg.mask_choose == 'full':
        arg.model_path = os.path.join(run_dir, arg.model_path)
        arg.decay_steps = len(data) // arg.batch_size
        train_features, train_labels, scaler_train,train_time= sequence_engineering_for_train(data.copy(),arg)
        model, evaluate_dict, model_config = train_model(train_time, train_features, train_labels, arg)
        print("数据量：", len(train_labels))
        print("model is fited!")
        print("full：完整 模型训练完毕！")

        # 存储评价指标：
        with open(os.path.join(arg.config_path, "info_full_{0}.json".format(arg.them)), mode='a', encoding='utf-8') as f:
            json.dump(evaluate_dict, f, indent=4, ensure_ascii=False)

    elif  arg.mask_choose == "predict":
        arg.model_path = os.path.join(arg.output_path,"run_"+arg.load_model_name[7:-27], arg.model_path)
        # 特征工程
        arg.decay_steps = len(data) // arg.batch_size
        data_features, data_labels, scaler_p,train_time = sequence_engineering_for_train(data.copy(),arg)

        # 加载模型
        model = load_model(arg.model_path, arg.load_model_name, arg.them)
        # 对于这种策略，这里不要
        data_last_time = data['time'].to_list()[-1]
        print("data 数据集合的最后一个日期：", data_last_time)

        # data_last_time = train_time[-1]
        # print("data 数据集合的最后一个日期：", data_last_time)
        # 预测未来的时间
        data = predict_feature_values(model,data_features,data_labels,scaler_p,data_last_time,arg)
    else:
        print("功能标志出错，需要重新定义mask_choose。")

# 第二种方式，每次都隔着2+1。
if __name__ == '__main__':
    enter()
