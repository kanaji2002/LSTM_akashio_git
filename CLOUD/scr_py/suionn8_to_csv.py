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


# スクレイピングを行う．
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import time


# 定期実行を行う．
import schedule

def scr_s():
        # Chromeのオプションを設定
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')

    # Chromeドライバーを初期化
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
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
    input_string = e1.text
    characters_to_remove = '℃'
    input_string = input_string.replace(characters_to_remove, '')
    driver.close()
    today_s=input_string
    return today_s


# # 待機
# time.sleep(10)
import csv
from datetime import datetime
def write_scr_data():
    

    # 今日の日付を取得
    today_date = datetime.now().strftime("%Y/%#m/%#d")

    # スクレイピングで取得したデータ（仮の値です）
    # scraped_data = {"value": 10.25}  # 実際のスクレイピング結果を入れてください
    scraped_data =scr_s()

    # CSVファイルのパス
    csv_file_path = "scr_csv/suionn-sum.csv"

    # CSVファイルを読み込み、既存のデータを取得
    existing_data = []
    with open(csv_file_path, 'r',encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        existing_data = list(reader)

    ## 今日の日付に対応するデータが存在するか確認
    today_data_index = None
    for i, data in enumerate(existing_data):
        # 今日と同じ日付を全探索で見つける（約7000件）
        if data.get("datetime") == today_date:
            #そのindex（0始まり）を保存
            today_data_index = i
            break
    if scraped_data=='欠測':
        # 今日の日付に対応するデータが存在する場合、スクレイピングで取得したデータで上書き
        scraped_data = existing_data[today_data_index-1]["suionn"]

    if today_data_index is not None:
        # 今日の日付に対応するデータが存在する場合、スクレイピングで取得したデータで上書き
        existing_data[today_data_index]["suionn"] = scraped_data

        updated_data=[existing_data[today_data_index]]
        # CSVファイルに書き込む
        header = ["datetime", "suionn"]
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            # # 対象の行だけを書き込む
            # for i, row in enumerate(existing_data):
            writer.writerows(existing_data)




        print(f"{today_date}のデータがCSVファイルに追加されました.ok ")
    else:
        print(f"{today_date}の日付が用意されていない可能性があります．")
    
    
    return 0

# write_scr_data()



#毎日01:30に実行
schedule.every().day.at("04:10").do(write_scr_data)

while True:
    # スケジュールに登録されたタスクを実行
    schedule.run_pending()
    # 1分ごとに確認
    time.sleep(60)



