import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates

#suionn2.py
# データの読み込み
# "water_temperature_data.csv"の代わりに"suionn-sum.csv"を使用
water_temp_df = pd.read_csv("suionn-sum.csv", parse_dates=['datetime'], dayfirst=True)
s_target = 'suionn'
data = water_temp_df[['datetime', s_target]]
data = data.set_index('datetime')  # 日付をインデックスに設定
dataset = data.values

# データを0〜1の範囲に正規化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

# 全体の80%をトレーニングデータとして扱う
training_data_len = int(np.ceil(len(dataset) * 0.8))

# どれくらいの期間をもとに予測するか
window_size = 60

train_data = scaled_data[0:int(training_data_len), :]

# train_dataをx_trainとy_trainに分ける
x_train, y_train = [], []
for i in range(window_size, len(train_data)):
    x_train.append(train_data[i-window_size:i, 0])
    y_train.append(train_data[i, 0])

# numpy arrayに変換
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# モデルの構築
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# モデルの学習
history = model.fit(x_train, y_train, batch_size=32, epochs=4, validation_split=0.1)

# エポックごとの損失と精度を取得
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']

# グラフにプロット
plt.plot(train_loss, label='Training Loss')
plt.plot(train_accuracy, label='Training Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# テストデータを作成
test_data = scaled_data[training_data_len - window_size:, :]

x_test = []
y_test = dataset[training_data_len:, :]
for i in range(window_size, len(test_data)):
    x_test.append(test_data[i-window_size:i, 0])

# numpy arrayに変換
x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# モデルによる予測
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

# 結果の可視化
train = data[:training_data_len]
valid = data[training_data_len:]
valid.loc[:, 'Predictions'] = predictions

# グラフに年だけを表示させる
fig, ax = plt.subplots(figsize=(16, 6))
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

plt.title('Water Temperature Prediction')
plt.xlabel('Year', fontsize=14)
plt.ylabel('Temperature', fontsize=14)
plt.plot(train[s_target])
plt.plot(valid[[s_target, 'Predictions']])
plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
plt.show()
valid.index = pd.to_datetime(valid.index)
# # 2022年10月15日の予測値と実際の値を取得
# prediction_2022_10_15 = valid.loc['2022/10/15', 'Predictions']
# actual_2022_10_15 = valid.loc['2022/10/15', s_target]

# 2022年10月14日の予測値と実際の値を取得
prediction_2022_10_14 = valid.loc['2022/10/14', 'Predictions']
actual_2022_10_14 = valid.loc['2022/10/14', s_target]
print(f"2022年10月14日の予測値: {prediction_2022_10_14}")
print(f"2022年10月14日の実測値: {actual_2022_10_14}")