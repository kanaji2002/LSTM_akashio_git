import torch
import torch.nn as nn
from torch.optim import SGD
import pandas as pd
import numpy as np

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size=inputDim,
                            hidden_size=hiddenDim,
                            batch_first=True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet(csv_file, data_length=50, freq=60., noise=0.00):
    """
    params\n
    csv_file : CSVファイルパス\n
    data_length : 各データの時系列長\n
    freq : 周波数\n
    noise : ノイズの振幅\n
    returns\n
    train_x : トレーニングデータ（t=1,2,...,size-1の値)\n
    train_t : トレーニングデータのラベル（t=sizeの値）\n
    """
    data = pd.read_csv(csv_file)
    train_x = []
    train_t = []

    for offset in range(len(data) - data_length):
        # 2つの説明変数を含むデータセットを作成
        x_values = data.iloc[offset:offset + data_length][['Tem', 'Sal']].values
        train_x.append([x_values + np.random.normal(loc=0.0, scale=noise) for _ in range(data_length)])
        
        # ラベルを作成
        label = data.iloc[offset + data_length]['label']
        train_t.append([label])

    return train_x, train_t

def mkRandomBatch(train_x, train_t, batch_size=10):
    """
    train_x, train_tを受け取ってbatch_x, batch_tを返す。
    """
    batch_x = []
    batch_t = []

    for _ in range(batch_size):
        idx = np.random.randint(0, len(train_x) - 1)
        batch_x.append(train_x[idx])
        batch_t.append(train_t[idx])
    
    return torch.tensor(batch_x), torch.tensor(batch_t)

def main():
    # CSVファイルのパスを指定
    csv_file = 'merged_data.csv'

    training_size = 10000
    test_size = 1000
    epochs_num = 1000
    hidden_size = 5
    batch_size = 100

    train_x, train_t = mkDataSet(csv_file)
    test_x, test_t = mkDataSet(csv_file)

    model = Predictor(2, hidden_size, 1)  # 説明変数が2つなのでinputDimを2に変更
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()

            data, label = mkRandomBatch(train_x, train_t, batch_size)

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            training_accuracy += np.sum(np.abs((output.detach().numpy() - label.detach().numpy()) < 0.1))

        # test
        test_accuracy = 0.0
        for i in range(int(test_size / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size]), torch.tensor(test_t[offset:offset+batch_size])
            output = model(data, None)

            test_accuracy += np.sum(np.abs((output.detach().numpy() - label.detach().numpy()) < 0.1))

        training_accuracy /= training_size
        test_accuracy /= test_size

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
