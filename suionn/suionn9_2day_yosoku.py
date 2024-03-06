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
#suionn9.py
# suionn7で予測するコードを関数化し，スクレイピングも組みこみ，ネットからの昨日のデータをもとに，今日，明日を予測する．


# # スクレイピングを行う．
# from selenium import webdriver
# from selenium.webdriver.common.by import By
# from selenium.webdriver.chrome.service import Service
# from selenium.webdriver.chrome.options import Options
# from selenium.webdriver.support.ui import WebDriverWait
# from selenium.webdriver.support import expected_conditions as EC
# from webdriver_manager.chrome import ChromeDriverManager
# import time


# # 定期実行を行う．
# import schedule

# def scr_s():
#         # Chromeのオプションを設定
#     options = Options()
#     options.add_argument('--no-sandbox')
#     options.add_argument('--disable-dev-shm-usage')

#     # Chromeドライバーを初期化
#     driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
#     driver.set_page_load_timeout(10)  # 10秒



#     try:
#         driver.get('https://suion.pref.kagawa.lg.jp/suion_hiuchi.php')
#         print("The page was loaded in time！")
#     except:
#         print("time out!")
        
#     # ターゲットのウェブページにアクセス

#     # driver.implicitly_wait(2)

#     # # 要素がクリック可能になるまで待機するWebDriverWaitを作成
#     # wait = WebDriverWait(driver, 5)

#     # XPathを使用して要素を見つける
#     e1 = driver.find_element(By.XPATH, "/html/body/div[3]/div[1]/div[2]/table/tbody/tr[1]/td[2]")

#     print(e1.text)
#     input_string = e1.text
#     characters_to_remove = '℃'
#     input_string = input_string.replace(characters_to_remove, '')

#     today_s=input_string
#     return today_s


# # # 待機
# # time.sleep(10)
# import csv
# from datetime import datetime
# def write_scr_data():
    

#     # 今日の日付を取得
#     today_date = datetime.now().strftime("%Y/%#m/%#d")

#     # スクレイピングで取得したデータ（仮の値です）
#     # scraped_data = {"value": 10.25}  # 実際のスクレイピング結果を入れてください
#     scraped_data =scr_s()
#     # CSVファイルのパス
#     csv_file_path = "suionn-sum.csv"

#     # CSVファイルを読み込み、既存のデータを取得
#     existing_data = []
#     with open(csv_file_path, 'r',encoding='utf-8-sig') as file:
#         reader = csv.DictReader(file)
#         existing_data = list(reader)

#     ## 今日の日付に対応するデータが存在するか確認
#     today_data_index = None
#     for i, data in enumerate(existing_data):
#         # 今日と同じ日付を全探索で見つける（約7000件）
#         if data.get("datetime") == today_date:
#             #そのindex（0始まり）を保存
#             today_data_index = i
#             break

#     if today_data_index is not None:
#         # 今日の日付に対応するデータが存在する場合、スクレイピングで取得したデータで上書き
#         existing_data[today_data_index]["suionn"] = scraped_data

#         updated_data=[existing_data[today_data_index]]
#         # CSVファイルに書き込む
#         header = ["datetime", "suionn"]
#         with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
#             writer = csv.DictWriter(file, fieldnames=header)
#             writer.writeheader()
#             # # 対象の行だけを書き込む
#             # for i, row in enumerate(existing_data):
#             writer.writerows(existing_data)




#         print(f"{today_date}のデータがCSVファイルに追加されました.ok ")
#     else:
#         print(f"{today_date}の日付が用意されていない可能性があります．")
        
#     return 0

# # write_scr_data()



# # 毎日01:30に実行
# schedule.every().day.at("18:47").do(write_scr_data)

# while True:
#     # スケジュールに登録されたタスクを実行
#     schedule.run_pending()
#     # 1分ごとに確認
#     time.sleep(60)








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


# 関数化して日付を引数にする．
# specified_dataは予測する日
# データは1時半に昨日のデータを取得する．そのデータから，今日の一日の平均水温を予測する．
# 日射量は毎日1時更新だが，水温は30分ごとに平均水温が更新される．そのため，水温は，最新のデータを使い，明日の予測だけでよくないか
# 日射量に関しては，昨日のデータをもとに，今日と明日のデータを予測する．
# つまり，今日の赤潮予測には，昨日までの日射量を用いた今日の日射量の予測値と，今日の平均水温（実測値）を使用する．

specified_data = '2024/3/6' # 翌日
specified_date = datetime.strptime(specified_data, '%Y/%m/%d')
specified_date_next = specified_date + timedelta(days=1)  # 翌々日
# データを0〜1の範囲に正規化
scaler = MinMaxScaler(feature_range=(0, 1))


if specified_data in data.index:

    
    # 指定した日を含まないように変更
    specified_date_index = data.index.get_loc(specified_data) - 1

    # テストデータを作成
    test_data = scaled_data[specified_date_index - window_size + 1:specified_date_index + 1, :]

    # 指定した日までのデータを使用して予測
    specified_data_for_model = scaled_data[specified_date_index - window_size:specified_date_index, :]  # size=60 正規化されている水温
    x_specified_date = np.reshape(specified_data_for_model, (1, specified_data_for_model.shape[0], 1)) #LSTMに入力する形式にしている．
    predicted_value_specified_date = model.predict(x_specified_date)
    #predicted_value_specified_date = scaler.inverse_transform(predicted_value_specified_date)


    # 予測された値と過去の実測値を結合してさらに次を予測
    #predicted_value_specified_date = scaler.fit_transform(predicted_value_specified_date)
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