import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from kan import KAN

# 1. 数据加载和预处理
data = pd.read_csv('../signal_metrics.csv')
data = data[['Timestamp', 'Network Type', 'Data Throughput (Mbps)']]
data['Timestamp'] = pd.to_datetime(data['Timestamp'])
data.set_index('Timestamp', inplace=True)
data = pd.get_dummies(data, columns=['Network Type'], drop_first=True)
data = data[data['Network Type_5G'] == True]
data.drop(columns=['Network Type_5G'], inplace=True)

data['Data Throughput (Mbps)'] = pd.to_numeric(data['Data Throughput (Mbps)'], errors='coerce')
data.dropna(inplace=True)

# 归一化
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)


# 2. 创建数据集（时间序列转换）
def create_sequences(data, seq_length=6):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i + seq_length])
        labels.append(data[i + seq_length])
    return np.array(sequences), np.array(labels)


seq_length = 6  # 以过去24小时的数据作为输入
X, y = create_sequences(data_scaled, seq_length)

# 数据集划分
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 转换为 PyTorch Tensor
X_train = torch.from_numpy(X_train).float()
X_test = torch.from_numpy(X_test).float()
y_train = torch.from_numpy(y_train).float()
y_test = torch.from_numpy(y_test).float()



# 3. 模型构建
class CNN_LSTM_KAN(nn.Module):
    def __init__(self, in_channels, hidden_size, num_layers, out_channels, output_size):
        super(CNN_LSTM_KAN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=1)
        )
        self.lstm = nn.LSTM(input_size=out_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.kan = KAN(width=[hidden_size, output_size])

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.kan(x[:, -1, :])
        return x


# 4. 训练模型
input_dim = 1
hidden_dim = 32
num_layers = 2
output_dim = 1
num_epochs = 100

# model = CNN_LSTM_KAN(in_channels=input_dim, hidden_size=hidden_dim, num_layers=num_layers, out_channels=32,
#                      output_size=output_dim)

model = CNN_LSTM_KAN(in_channels=3, hidden_size=hidden_dim, num_layers=num_layers, out_channels=32, output_size=output_dim)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)


def train_model(model, X_train, y_train, X_test, y_test, num_epochs=100):
    best_loss = float('inf')
    patience, patience_counter = 10, 0

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            val_outputs = model(X_test)
            val_loss = criterion(val_outputs, y_test)

        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}')

        # # 早停机制
        # if val_loss.item() < best_loss:
        #     best_loss = val_loss.item()
        #     patience_counter = 0
        # else:
        #     patience_counter += 1
        #
        # if patience_counter >= patience:
        #     print("Early stopping triggered.")
        #     break


train_model(model, X_train, y_train, X_test, y_test, num_epochs=num_epochs)

# 5. 预测
model.eval()
with torch.no_grad():
    y_pred = model(X_test)

# 反归一化
# y_test = scaler.inverse_transform(y_test.numpy().reshape(-1, 1))
# y_pred = scaler.inverse_transform(y_pred.numpy().reshape(-1, 1))

# 打印预测结果
print("真实值 vs 预测值:")
for real, pred in zip(y_test[:10], y_pred[:10]):
    print(f"Actual: {real[0]:.2f}, Predicted: {pred[0]:.2f}")
