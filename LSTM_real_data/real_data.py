import torch
import torch.nn as nn
from torch.optim import SGD
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
# real_data.py
class Predictor(nn.Module):
    def __init__(self, inputDim, hiddenDim, outputDim):
        super(Predictor, self).__init__()

        self.output_layer = nn.Linear(hiddenDim, outputDim)
    
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)
        output = self.output_layer(output[:, -1, :])

        return output

def mkDataSet(csv_file, label_column, test_size=0.1, data_length=50):
    data = pd.read_csv(csv_file)

    # Extract the relevant columns for input data and labels
    # input_columns = ["Tem", "DO","Sal","nissyaryou"]
    input_columns = ["Tem","Sal","nissyaryou"]
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
    csv_file_path = "data/edited_akashio_data/HIU_data_+n.csv"  # Replace with the path to your CSV file
    label_column = "Chl.a"  # Replace with the actual column name for the label

    train_x, train_t, test_x, test_t = mkDataSet(csv_file_path, label_column)

    # Adjust input and output dimensions based on the columns used
    input_dim = len(train_x[0][0])  # Number of features in the input
    output_dim = 1  # Since we are predicting a single value

    hidden_size = 5
    epochs_num = 2000
    batch_size = 100

    model = Predictor(input_dim, hidden_size, output_dim)
    criterion = nn.MSELoss()
    optimizer = SGD(model.parameters(), lr=0.01)
    training_losses = []
    training_accuracies = []
    test_accuracies = []
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
            training_accuracy += torch.sum(torch.lt(torch.abs(output.detach() - label), 500).float())

        # test
        test_accuracy = 0.0
        for i in range(int(len(test_x) / batch_size)):
            offset = i * batch_size
            data, label = torch.tensor(test_x[offset:offset+batch_size], dtype=torch.float32), torch.tensor(test_t[offset:offset+batch_size], dtype=torch.float32).view(-1, 1)
            output = model(data, None)

            test_accuracy += torch.sum(torch.lt(torch.abs(output.detach() - label), 500).float())

        training_accuracy /= len(train_x)
        test_accuracy /= len(test_x)
        
        training_losses.append(running_loss)
        training_accuracies.append(training_accuracy.item())
        test_accuracies.append(test_accuracy)

        print('%d loss: %.3f, training_accuracy: %.5f, test_accuracy: %.5f' % (
            epoch + 1, running_loss, training_accuracy, test_accuracy))
        
    plt.figure(figsize=(10, 5))
    plt.plot(training_losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plotting the training and test accuracies
    plt.figure(figsize=(10, 5))
    plt.plot(training_accuracies, label='Training Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Training and Test Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    main()
