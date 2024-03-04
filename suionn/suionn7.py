from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import Model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import tensorflow
from tensorflow.python.keras.models import load_model
#suionn7.py
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
model = ()
model=load_model('model_from_suionn6.h5')
# ここから変更--------
specified_data = '2021/11/28'
# テストデータを作成
test_data = scaled_data[training_data_len - window_size:, :]

# 指定した日までのデータを使用して予測
specified_date_index = data.index.get_loc(specified_data)
specified_data_for_model = scaled_data[specified_date_index - window_size:specified_date_index, :]
x_specified_date = np.reshape(specified_data_for_model, (1, specified_data_for_model.shape[0], 1))
predicted_value_specified_date = model.predict(x_specified_date)
predicted_value_specified_date = scaler.inverse_transform(predicted_value_specified_date)

# 予測された値と過去の実測値を結合してさらに次を予測
combined_data_specified_date = np.concatenate((specified_data_for_model, predicted_value_specified_date), axis=0)
next_prediction_data_specified_date = combined_data_specified_date[-window_size:, :]
x_next_specified_date = np.reshape(next_prediction_data_specified_date, (1, next_prediction_data_specified_date.shape[0], 1))
next_prediction_specified_date = model.predict(x_next_specified_date)
next_prediction_specified_date = scaler.inverse_transform(next_prediction_specified_date)

# 1日進めて次の日を予測
specified_date_next_data = scaled_data[specified_date_index:specified_date_index + 1, :]
x_specified_date_next = np.reshape(specified_date_next_data, (1, specified_date_next_data.shape[0], 1))
predicted_value_specified_date_next = model.predict(x_specified_date_next)
predicted_value_specified_date_next = scaler.inverse_transform(predicted_value_specified_date_next)


specified_date = datetime.strptime(specified_data, '%Y/%m/%d')
specified_date_next = specified_date + timedelta(days=1)  # 1日進める
# 結果の表示
print(f"predict {specified_data}: {predicted_value_specified_date[-1, 0]}")
print(f"predict {specified_date_next}: {predicted_value_specified_date_next[-1, 0]}")
