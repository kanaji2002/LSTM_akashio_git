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
#suionn8.py
# suionn7で予測するコードを関数化し，スクレイピングも組みこみ，ネットからの昨日のデータをもとに，今日，明日を予測する．



from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time
# selenium3.py


def scr_s():
        # Chromeのオプションを設定
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Chromeドライバーを初期化
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    driver.set_page_load_timeout(10)  # 10秒



    try:
        driver.get('https://suion.pref.kagawa.lg.jp/suion_hiuchi.php')
        print("The page was loaded in time！")
    except:
        print("time out!")
        
    # ターゲットのウェブページにアクセス

    # driver.implicitly_wait(2)

    # # 要素がクリック可能になるまで待機するWebDriverWaitを作成
    # wait = WebDriverWait(driver, 5)

    # XPathを使用して要素を見つける
    e1 = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[2]/table/tbody/tr[1]/td[2]")

    print(e1.text)
    today_s=e1.text
    return today_s


# # 待機
# time.sleep(10)








# データの読み込み
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







specified_data = '2024/3/6'
specified_date = datetime.strptime(specified_data, '%Y/%m/%d')
specified_date_next = specified_date + timedelta(days=1)  # 1日進める
# データを0〜1の範囲に正規化
scaler = MinMaxScaler(feature_range=(0, 1))


if specified_data in data.index:

    
    # 指定した日を含まないように変更
    specified_date_index = data.index.get_loc(specified_data) - 1

    # テストデータを作成
    test_data = scaled_data[specified_date_index - window_size + 1:specified_date_index + 1, :]

    # 指定した日までのデータを使用して予測
    specified_data_for_model = scaled_data[specified_date_index - window_size:specified_date_index, :]
    x_specified_date = np.reshape(specified_data_for_model, (1, specified_data_for_model.shape[0], 1))
    predicted_value_specified_date = model.predict(x_specified_date)
    # predicted_value_specified_date = scaler.inverse_transform(predicted_value_specified_date)

    # 予測された値と過去の実測値を結合してさらに次を予測
    # predicted_value_specified_date = scaler.fit_transform(predicted_value_specified_date)
    combined_data_specified_date = np.concatenate((specified_data_for_model, predicted_value_specified_date), axis=0)
    next_prediction_data_specified_date = combined_data_specified_date[-window_size:, :]
    x_next_specified_date = np.reshape(next_prediction_data_specified_date, (1, next_prediction_data_specified_date.shape[0], 1))
    next_prediction_specified_date = model.predict(x_next_specified_date)
    # next_prediction_specified_date = scaler.inverse_transform(next_prediction_specified_date)
    print(f'combined_data_specified_date{combined_data_specified_date}')
    # 1日進めて次の日を予測
    specified_date_next_data = scaled_data[specified_date_index + 1:specified_date_index + 2, :]
    x_specified_date_next = np.reshape(specified_date_next_data, (1, specified_date_next_data.shape[0], 1))
    predicted_value_specified_date_next = model.predict(x_specified_date_next)
    predicted_value_specified_date_next = scaler.inverse_transform(predicted_value_specified_date_next)

    # 結果の表示
    print(f"predict {specified_data}: {predicted_value_specified_date[-1, 0]}")
    print(f"predict {specified_date_next}: {predicted_value_specified_date_next[-1, 0]}")
    
else:
    print(f"The specified date '{specified_data}' does not exist in the data.")