import pandas as pd

# HIU_data_all_data.csvを読み込む
hiu_data = pd.read_csv('data/edited_akashio_data/HIU_data_all_data.csv')

# 降水量のCSVファイルを読み込む
precipitation_data = pd.read_csv('tyouryuu/get_uv_250m_csv/merge.csv')

# 日付と時刻の列をdatetime型に変換
hiu_data['datetime'] = pd.to_datetime(hiu_data['datetime']) + pd.to_timedelta(hiu_data['hour'], unit='h')

# 降水量データの日付と時刻の列をdatetime型に変換
precipitation_data['datetime'] = pd.to_datetime(precipitation_data['datetime'])

# 月、日、時刻の列を抽出して新しい列を作成
hiu_data['month'] = hiu_data['datetime'].dt.month
hiu_data['day'] = hiu_data['datetime'].dt.day
hiu_data['time'] = hiu_data['datetime'].dt.time

precipitation_data['month'] = precipitation_data['datetime'].dt.month
precipitation_data['day'] = precipitation_data['datetime'].dt.day
precipitation_data['time'] = precipitation_data['datetime'].dt.time

# 月、日、時刻をキーとしてマージ
hiu_data = pd.merge(hiu_data, precipitation_data, on=['month', 'day', 'time'], how='left', suffixes=('', '_1pre'))

# 変更を保存する
hiu_data.to_csv('data/edited_akashio_data/HIU_data_all_data2.csv', index=False)