import csv
import pandas as pd
from datetime import datetime, timedelta
import math
from dateutil.relativedelta import relativedelta

def read_csv_and_extract_uv(csv_file,new_date):
    data = pd.DataFrame(columns=['date', 'u', 'v', 'uv'])

    
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            new_date = datetime.strptime(new_date, "%Y/%m/%d %H:%M")
            new_date = (new_date + timedelta(hours=1)).strftime("%Y/%m/%d %H:%M")
            # 条件を満たす行のデータのみ処理
            
            for _ in range(65):
                try:
                    row = next(reader)
                except StopIteration:
                    break  # 次の行がない場合はループを抜ける
            
            
            try:
                u_value = float(row[39])  # uの値を取得
            except (ValueError, IndexError):
                continue  # 数値変換エラーまたはインデックスエラーが発生した場合は次の行に進む

            for _ in range(186):
                try:
                    row = next(reader)
                except StopIteration:
                    break  # 次の行がない場合はループを抜ける    

            try:
                v_value = float(row[38])  # vの値を取得
            except (ValueError, IndexError):
                continue  # 数値変換エラーまたはインデックスエラーが発生した場合は次の行に進む
            
            uv_value = math.sqrt(u_value**2 + v_value**2)
                    
            # 新しい行をデータに追加
            df2 = pd.DataFrame({'date': [new_date], 'u': [u_value], 'v': [v_value], 'uv': [uv_value]})
            data = pd.concat([data, df2])
            
            for _ in range(120):
                try:
                    row = next(reader)
                except StopIteration:
                    break  # 次の行がない場合はループを抜ける    

    # print(data)
    return data


# 初期の日付を設定します
initial_date = datetime(2015,12,31,23,00)

for i in range(0,12):
    input_file = f'tyouryuu/prepro_250m_csv/edited_{i+1:02d}.csv'
    output_file = f'tyouryuu/get_uv_250m_csv/get_uv_{i+1:02d}.csv'

    # 新しい日付を計算します
    new_date = initial_date + relativedelta(months=i)
    
    #read_csv_and_extract_uv()が文字列で処理を行っていた（関数内でdatetime型に直していた）ので，文字型にしておく（2度手間ではある）
    date_format = "%Y/%m/%d %H:%M"
    date_string = new_date.strftime(date_format)

    

    data = read_csv_and_extract_uv(input_file,date_string)

    # データをデータフレームに変換して保存する
    df = pd.DataFrame(data)
    df.to_csv(output_file, index=False)  # index=False を指定して行番号を保存しないようにする

