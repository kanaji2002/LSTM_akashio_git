from tensorflow.keras.models import Sequential
# from tensorflow.python.keras.layers import Dense, LSTM, Dropout
from tensorflow.python.keras.models import Model

from tensorflow.keras.layers import LSTM
import keras
from keras.layers import Dense
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as dates
from datetime import datetime, timedelta
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import load_model
#suionn9.py


# データの読み込み
water_temp_df = pd.read_csv("nissyaryou/nissyaryou_takamatsu.csv", parse_dates=['datetime'], dayfirst=True)
s_target = 'takamatsu'
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
# window_size=60
# model=load_model('nissyaryou/model_from_nissayryou_model_save.h5')

# window_size=365
model=load_model('nissyaryou/model_from_nissayryou_model_save_win365.h5')

#nissyaryou\model_from_nissayryou_model_save.h5
# ここから変更--------


# 関数化して日付を引数にする．
# specified_dataは予測する日
# データは1時半に昨日のデータを取得する．そのデータから，今日の一日の平均水温を予測する．
# 日射量は毎日1時更新だが，水温は30分ごとに平均水温が更新される．そのため，水温は，最新のデータを使い，明日の予測だけでよくないか
# 日射量に関しては，昨日のデータをもとに，今日と明日のデータを予測する．
# つまり，今日の赤潮予測には，昨日までの日射量を用いた今日の日射量の予測値と，今日の平均水温（実測値）を使用する．

# specified_data = '2024/3/6' # 翌日
# specified_date = datetime.strptime(specified_data, '%Y/%#m/%#d')
# specified_date_next = specified_date + timedelta(days=1)  # 翌々日
# データを0〜1の範囲に正規化
#デフォで0-1の正規化をするので，feature_range=(0, 1)は書かなくていい



def predict_temperature(model, specified_data):
    # specified_date = datetime.strptime(specified_data, '%Y/%m/%d')
    print('format')
    print(f'today : {specified_data}')

    # print("データの最初の日付:", data.index.min())
    # print("データの最後の日付:", data.index.max())

    # # 日付のフォーマット
    # specified_date = '2002/1/1'
    # specified_date_format = datetime.strptime(specified_date, '%Y/%m/%d')
    # print("指定された日付のフォーマット:", specified_date_format)


    if specified_date in data.index:
        specified_date_index = data.index.get_loc(specified_date) - 1

        for day in range(30):
            # テストデータを作成
            test_data = scaled_data[specified_date_index - window_size + 1:specified_date_index + 1, :]

            # 指定した日までのデータを使用して予測
            specified_data_for_model = scaled_data[specified_date_index - window_size:specified_date_index, :]
            x_specified_date = np.reshape(specified_data_for_model, (1, specified_data_for_model.shape[0], 1))
            predicted_value_specified_date = model.predict(x_specified_date)

            # 予測された値と過去の実測値を結合してさらに次を予測
            combined_data_specified_date = np.concatenate((specified_data_for_model, predicted_value_specified_date), axis=0)
            # 予測データが結合された後に直近のデータを60個取った
            next_prediction_data_specified_date = combined_data_specified_date[-window_size:, :]
            next_prediction_data_specified_date_reshaped = np.reshape(next_prediction_data_specified_date, (window_size, 1))
            scaled_data[specified_date_index - window_size+day+1:specified_date_index + day+1 , :] = next_prediction_data_specified_date_reshaped            # next_prediction_specified_date = model.predict(x_next_specified_date)

            predicted_value_specified_date = scaler.inverse_transform(predicted_value_specified_date)
            # next_prediction_specified_date = scaler.inverse_transform(next_prediction_specified_date)

            # 結果の表示
            if day == 0:
                current_date = specified_date
                print(f"Predicted temperature for {current_date}: {predicted_value_specified_date[-1, 0]}")
                specified_date_next=datetime.strptime(specified_date, '%Y/%m/%d')
                # next_date = specified_date_next + timedelta(days=day+1)
                # next_date_srt=datetime.strftime(next_date, '%Y/%#m/%#d')
                # specified_date_next=datetime.strptime(specified_date, '%Y/%m/%d')
                # print(f"Predicted temperature for {next_date_srt}: {predicted_value_specified_date[-1, 0]}")
            
            else:
                
                next_date = datetime.strptime(specified_date, '%Y/%m/%d') + timedelta(days=day)
                
                # next_date=datetime.strftime(next_date, '%Y/%#m/%#d')
                next_date_srt=datetime.strftime(next_date, '%Y/%#m/%#d')
                print(f"Predicted temperature for {next_date_srt}: {predicted_value_specified_date[-1, 0]}")

            specified_date_index += 1

    else:
        print(f"The specified date '{specified_data}' does not exist in the data.")







# 強制的な日付指定
specified_date = '2024/3/12'

#specified_date = datetime.now().strftime("%Y/%#m/%#d")
predict_temperature(model,specified_date)  # 例として指定した日の予測を実行

