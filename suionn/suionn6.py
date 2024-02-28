import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import matplotlib.dates as mdates
# ある指定した日Xまでの実測値を用いて，予測値を出し，その予測値と，Xまでの実測値でさらに予測値を出す．
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
history = model.fit(x_train, y_train, batch_size=32, epochs=4)

# テストデータを作成
test_data = scaled_data[training_data_len - window_size:, :]


# ここから変更--------
specified_data='2022/5/22'
# 特定の日のインデックスを取得
specified_date_index = data.index.get_loc(specified_data)  # '指定した日の日付' には実際の日付を指定してください

# 過去のデータを取得
past_data = scaled_data[specified_date_index - window_size:specified_date_index, :]

# 過去のデータを使って予測
x_past = []
for i in range(window_size, len(past_data)):
    x_past.append(past_data[i - window_size:i, 0])

# numpy arrayに変換
x_past = np.array(x_past)
x_past = np.reshape(x_past, (x_past.shape[0], x_past.shape[1], 1))

# モデルによる予測
predicted_value = model.predict(x_past)
predicted_value = scaler.inverse_transform(predicted_value)

# 予測された値と過去の実測値を結合してさらに次を予測
combined_data = np.concatenate((past_data, predicted_value), axis=0)
next_prediction_data = combined_data[-window_size:, :]

# numpy arrayに変換
x_next = np.reshape(next_prediction_data, (1, next_prediction_data.shape[0], 1))

# モデルによるさらなる予測
next_prediction = model.predict(x_next)
next_prediction = scaler.inverse_transform(next_prediction)

print(f"指定した日の予測値: {predicted_value[-1, 0]}")
print(f"指定した日の次の予測値: {next_prediction[0, 0]}")






















# x_test = []
# y_test = dataset[training_data_len:, :]
# for i in range(window_size, len(test_data)):
#     x_test.append(test_data[i-window_size:i, 0])

# # numpy arrayに変換
# x_test = np.array(x_test)
# x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# # モデルによる予測
# predictions = model.predict(x_test)
# predictions = scaler.inverse_transform(predictions)

# # 結果の可視化
# train = data[:training_data_len]
# valid = data[training_data_len:]
# valid.loc[:, 'Predictions'] = predictions

# # グラフに年だけを表示させる
# fig, ax = plt.subplots(figsize=(16, 6))
# ax.xaxis.set_major_locator(mdates.YearLocator())
# ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

# plt.title('Water Temperature Prediction')
# plt.xlabel('Year', fontsize=14)
# plt.ylabel('Temperature', fontsize=14)
# plt.plot(train[s_target])
# plt.plot(valid[[s_target, 'Predictions']])
# plt.legend(['Train', 'Actual', 'Predictions'], loc='lower right')
# plt.show()
# valid.index = pd.to_datetime(valid.index)
# # 2022年10月15日の予測値と実際の値を取得
# prediction_2022_10_15 = valid.loc['2022/10/15', 'Predictions']
# actual_2022_10_15 = valid.loc['2022/10/15', s_target]

# print(f"2022年10月15日の予測値: {prediction_2022_10_15}")
# print(f"2022年10月15日の実測値: {actual_2022_10_15}")