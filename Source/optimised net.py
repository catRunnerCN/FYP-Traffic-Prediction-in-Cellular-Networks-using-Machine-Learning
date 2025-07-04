import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, BatchNormalization, Bidirectional, LSTM, GRU
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import ReduceLROnPlateau

# 构建模型
model = Sequential()

# 第一层 CNN (使用 Batch Normalization 和 Dilated Convolution)
model.add(Conv1D(64, kernel_size=3, dilation_rate=2, activation='relu', input_shape=(480, 1)))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# 第二层 CNN
model.add(Conv1D(128, kernel_size=3, dilation_rate=2, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# 第三层 CNN
model.add(Conv1D(256, kernel_size=3, activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling1D(pool_size=2))

# BiLSTM 层
model.add(Bidirectional(LSTM(100, return_sequences=True)))
model.add(Dropout(0.3))

# GRU 代替第二层 LSTM，计算更高效
model.add(GRU(50))
model.add(Dropout(0.3))

# 全连接层
model.add(Dense(64, activation='swish'))
model.add(Dropout(0.3))
model.add(Dense(32, activation='swish'))
model.add(Dense(1, activation='sigmoid'))  # 二分类问题，回归问题可改为 'linear'

# 编译模型
optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# 训练策略
lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)

# 输出模型结构
model.summary()
