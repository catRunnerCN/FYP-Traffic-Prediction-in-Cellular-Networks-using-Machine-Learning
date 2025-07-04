import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
# # 加载原始数据
# data = pd.read_csv('../signal_metrics.csv')
#
# # 选择需要的列：Timestamp 和 Data Throughput (Mbps)
# data = data[['Timestamp', 'Network Type', 'Data Throughput (Mbps)']]
# # 确保 Timestamp 列是时间格式
# data['Timestamp'] = pd.to_datetime(data['Timestamp'])
#
# # 设置 Timestamp 为索引
# data.set_index('Timestamp', inplace=True)
#
# data = pd.get_dummies(data, columns=['Network Type'], drop_first=True)
#
# # 查看数据的前几行
# print(data.head())