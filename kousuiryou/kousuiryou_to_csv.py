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

# nissyaryou/nissyaryou_to_csv.py
def scr_s():
        # Chromeのオプションを設定
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')


    # Chromeドライバーを初期化
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    driver.set_page_load_timeout(10)  # 10秒



    try:
        driver.get('https://www.data.jma.go.jp/gmd/risk/obsdl/')
        print("The page was loaded in time！")
    except:
        print("time out!")
        
    # ターゲットのウェブページにアクセス

    driver.implicitly_wait(2)
    wait = WebDriverWait(driver, 10)
    # # 要素がクリック可能になるまで待機するWebDriverWaitを作成
    # wait = WebDriverWait(driver, 5)

    # XPathを使用して要素を見つける
    # XPathを使用して要素を見つける
    e1 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/table/tbody/tr[14]/td[7]/div")
    #/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/table/tbody/tr[14]/td[7]/div
    # 要素をクリック
    e1.click()


    # 高松を選択
    e2 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[1]/div[2]/div[3]/div")
    # 要素をクリック
    e2.click()

    # 項目を選ぶ
    e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[1]/img[2]")
    # 要素をクリック
    e3.click()

    # 日照/日射
    e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div[3]/div[1]/ul/li[2]/a")
    # 要素をクリック
    e3.click()

    # 日合計全天日射量を選択
    e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[2]/div[3]/div[1]/div[2]/table/tbody/tr[1]/td/span/label")
    # 要素をクリック
    e3.click()

    # 期間を選ぶ
    e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[1]/img[3]")
    # 要素をクリック
    e3.click()

    # 最近一か月
    e3 = driver.find_element(By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[1]/div[2]/div[3]/div/div/div[1]/div[2]/div[1]/input[2]")
    # 要素をクリック
    e3.click()


    # 表示
    e3 = wait.until(EC.visibility_of_element_located((By.XPATH, "/html/body/div[1]/div[2]/div[1]/div[2]/div[1]/div/div[2]/div[2]/div/div/div[1]/div[1]/span/img")))
    # 要素をクリック
    e3.click()

    #textを取得する
    text = wait.until(EC.visibility_of_element_located((By.XPATH, '/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[5]/div[3]/div/div[30]/div/span'))).text
                                        #/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[5]/div[3]/div/div[1]/div/span
    #/html/body/div[1]/div[2]/div[1]/div[2]/div[2]/div[2]/div[5]/div[3]/div/div[20]/div
    print(text)
    input_string = text
    # characters_to_remove = '℃'
    # input_string = input_string.replace(characters_to_remove, '')
    driver.close()
    today_n=input_string   #today_nは，todayのnissayryou
    return today_n


# # 待機
# time.sleep(10)
import csv
from datetime import datetime,timedelta



def write_scr_data():
    

    # 今日の日付を取得
    # today_date = datetime.now().strftime("%Y/%#m/%#d")
    yest_date=datetime.now() - timedelta(days=1)
    yest_date=yest_date.strftime("%Y/%#m/%#d")
    print(yest_date)

    # スクレイピングで取得したデータ（仮の値です）
    # scraped_data = {"value": 10.25}  # 実際のスクレイピング結果を入れてください
    scraped_data =scr_s()
    # CSVファイルのパス
    csv_file_path = "kousuiryou/kousuiryou.csv"

    # CSVファイルを読み込み、既存のデータを取得
    existing_data = []
    with open(csv_file_path, 'r',encoding='utf-8-sig') as file:
        reader = csv.DictReader(file)
        existing_data = list(reader)

    ## 機能の日付に対応するデータが存在するか確認
    yest_data_index = None
    for i, data in enumerate(existing_data):
        # 昨日と同じ日付を全探索で見つける（約7000件）
        if data.get("datetime") == yest_date:
            #そのindex（0始まり）を保存
            yest_data_index = i
            break

    if yest_data_index is not None:
        # 今日の日付に対応するデータが存在する場合、スクレイピングで取得したデータで上書き
        existing_data[yest_data_index]["kousuiryou"] = scraped_data

        # updated_data=[existing_data[yest_data_index]]
        # CSVファイルに書き込む
        header = ["datetime", "kousuiryou"]
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=header)
            writer.writeheader()
            # # 対象の行だけを書き込む
            # for i, row in enumerate(existing_data):
            writer.writerows(existing_data)




        print(f"{yest_date}のデータがCSVファイルに追加されました.ok ")
    else:
        print(f"{yest_date}の日付が用意されていない可能性があります．")
        
    return 0

write_scr_data()



# # 毎日01:30に実行
# schedule.every().day.at("13:47").do(write_scr_data)

# while True:
#     # スケジュールに登録されたタスクを実行
#     schedule.run_pending()
#     # 1分ごとに確認
#     time.sleep(60)




