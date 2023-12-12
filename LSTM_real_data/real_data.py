import torch
import torch.nn as nn
from torch.optim import SGD
import pandas as pd
from sklearn.model_selection import train_test_split

class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.rnn = nn.LSTM(input_size=inputDim, hidden_size=hiddenDim, batch_first=True)
        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet(csv_file, label_column, test_size=0.1, data_length=50):
    data = pd.read_csv(csv_file)

    # Extract the relevant columns for input data and labels
    input_columns = ["Tem", "DO","Sal","nissyaryou",]
    input_data = data[input_columns]
    labels = data[label_column]

    # Split the data into training and test sets
    train_data, test_data, train_labels, test_labels = train_test_split(input_data, labels, test_size=test_size, shuffle=True, random_state=42)

    # Convert data to lists of lists (sequences)
    train_x = [train_data.iloc[i:i + data_length].values.tolist() for i in range(len(train_data) - data_length)]
    train_t = [train_labels.iloc[i + data_length] for i in range(len(train_data) - data_length)]

    test_x = [test_data.iloc[i:i + data_length].values.tolist() for i in range(len(test_data) - data_length)]
    test_t = [test_labels.iloc[i + data_length] for i in range(len(test_data) - data_length)]

    return train_x, train_t, test_x, test_t

def main():
    csv_file_path = "C:/Users/Kanaji Rinntarou/OneDrive - 独立行政法人 国立高等専門学校機構/Desktop/kennkyuu/LSTM_akashio/LSTM_real_data/merged_data_not_nan.csv"  # Replace with the path to your CSV file
    label_column = "Chl.a"  # Replace with the actual column name for the label

    train_x, train_t, test_x, test_t = mkDataSet(csv_file_path, label_column)

    # Adjust input and output dimensions based on the columns used
    input_dim = len(train_x[0][0])  # Number of features in the input
    output_dim = 1  # Since we are predicting a single value

    hidden_size = 5
    epochs_num = 1000
    batch_size = 100

    model = Predictor(input_dim, hidden_size, output_dim)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)

    for epoch in range(epochs_num):
        # training
       # training
        # training
       # training
        running_loss = 0.0
        training_accuracy = 0.0
        for i in range(int(len(train_x) / batch_size)):
            optimizer.zero_grad()

            data, label = torch.tensor(train_x[i * batch_size:(i + 1) * batch_size], dtype=torch.float32), torch.tensor(train_t[i * batch_size:(i + 1) * batch_size], dtype=torch.float32).view(-1, 1)

            output = model(data)

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            # しきい値未満かどうかを判定してから浮動小数点数に変換
            training_accuracy += torch.sum(torch.lt(torch.abs(output.detach() - label), 0.1).float())

        # test
        test_accuracy = 0.0
        for i in range(int(len(test_x) / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size], dtype=torch.float32), torch.tensor(test_t[offset:offset+batch_size], dtype=torch.float32).view(-1, 1)
            output = model(data, None)

            test_accuracy += torch.sum(torch.lt(torch.abs(output.detach() - label), 0.1).float())

        training_accuracy /= len(train_x)
        test_accuracy /= len(test_x)

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))

if __name__ == '__main__':
    main()
