import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# CSVファイルのパス
filepath = 'LSTM_akashio_output/CSV/number.csv'

# CSVファイルからデータを読み込み
aka = pd.read_csv(filepath, delimiter=';')

# 説明変数と目的変数を分割
X = aka.iloc[:, :-1]  # 最後の列を除くすべての列を説明変数とする
y = aka.iloc[:, -1]   # 最後の列を目的変数とする

# データの正規化
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X = torch.tensor(X_normalized, dtype=torch.float32)
y = torch.tensor(y.values, dtype=torch.float32)  # yをNumPy配列から取得

# LSTMモデル定義
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return torch.sigmoid(out)

# モデル、損失関数、最適化手法の定義
input_size = X.shape[1]
hidden_size = 300
output_size = 1

model = LSTMModel(input_size, hidden_size, output_size)
criterion = nn.BCEWithLogitsLoss()
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# モデルの学習
num_epochs = 30
epochs_list = []
loss_list = []

for epoch in range(num_epochs):
    outputs = model(X.view(-1, 1, input_size))
    loss = criterion(outputs.view(-1), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch == 0 or (epoch % 10 == 9):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
        epochs_list.append(epoch)
        loss_list.append(loss.item())

# 損失関数のラベルの指定と表示
plt.plot(epochs_list, loss_list, color="k")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss vs. Epoch')
plt.show()

# モデルの評価
model.eval()
with torch.no_grad():
    # 学習データに対する予測
    train_output = model(X.view(-1, 1, input_size))
    train_predicted = (train_output.view(-1).numpy() > 0.5).astype(int)

    # 学習データの予測結果をグラフに描画
    plt.figure(figsize=(10, 5))
    plt.plot(y.numpy(), label='Actual labels', marker='o')
    plt.plot(train_predicted, label='Predicted labels', marker='x')
    plt.title('Training Data: Actual vs. Predicted Labels')
    plt.xlabel('Sample')
    plt.ylabel('Label')
    plt.legend()
    plt.show()

    # 新しいデータに対する予測
    new_data = torch.tensor([[3, 43, 2, 13,3,3,2,4,5,6,22]], dtype=torch.float32)
    test_output = model(new_data.view(-1, 1, input_size))
    test_predicted = (test_output.view(-1).numpy() > 0.5).astype(int)

    # 新しいデータの予測結果を表示
    print("Predicted labels for new data:", test_predicted)
