import pandas as pd
import os

# 空のDataFrameを作成
result_data = pd.DataFrame(columns=['kai', 'year', 'mon', 'day', 'hour', 'minute', 'Tem','Sal','DO','Chl.a'])

# ファイルパスのベース
file_base_path = 'akashio-data/{}.csv'

# 各年のデータを処理
for year in range(2000, 2023):
    # ファイルパスを作成
    file_path = file_base_path.format(year)

    data = pd.read_csv(file_path, encoding='shift_jis')
    data['kai']=data['kai'].str.upper()
    # HIUのkaiに対応するデータを抽出
    hiu_data = data[data['kai'] == 'HIU']

    # 'Chl.a'のうち、手前にあるものを抽出
    selected_columns = ['kai', 'year', 'mon', 'day', 'time','Tem','Sal','DO', 'Chl.a']
    selected_data = hiu_data[selected_columns].reset_index(drop=True)
    selected_data = selected_data.replace('-', pd.NA,).dropna()
    selected_data['Chl.a']=selected_data['Chl.a'].str.replace('>','')
    selected_data['Chl.a']=selected_data['Chl.a'].str.replace('<','')
    
    # 'time'列を時、分、秒に分割して数値化
    time_split = selected_data['time'].str.split(':', expand=True)
    # 'hour'と'minute'を新しい列として追加
    selected_data['hour'] = time_split[0].astype(int)
    selected_data['minute'] = time_split[1].astype(int)
    # 'day'列を数値化
    selected_data['day'] = pd.to_numeric(selected_data['day'], errors='coerce').dropna().astype(int)
    # 'time'列を削除
    selected_data.drop('time', axis=1, inplace=True)

    # 「-」や空白を含む行を削除
    selected_data = selected_data.replace('-', pd.NA).dropna()

    result_data = pd.concat([result_data, selected_data], ignore_index=True)

# 時系列順にソート
result_data.sort_values(['year', 'mon', 'day', 'hour', 'minute'], inplace=True)

result_data.to_csv('edited_akashio_data/HIU_data.csv', index=False)

# 結果を表示
print(result_data)
