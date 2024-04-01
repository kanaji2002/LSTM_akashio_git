import pandas as pd

# HIU_data_all_data.csvを読み込む
hiu_data = pd.read_csv('data/edited_akashio_data/HIU_data_all_data.csv')

# 降水量のCSVファイルを読み込む（仮のファイル名としてprecipitation.csvを想定）
precipitation_data = pd.read_csv('kousuiryou/kousuiryou.csv')

# 日付と時刻の列をdatetime型に変換
hiu_data['datetime'] = pd.to_datetime(hiu_data['datetime'])

# 降水量データの日付と時刻の列をdatetime型に変換
precipitation_data['datetime'] = pd.to_datetime(precipitation_data['datetime'])

# 日付を1日ずらす
precipitation_data['datetime'] = precipitation_data['datetime'] + pd.DateOffset(days=1)

# HIU_data_all_data.csvの日付と時刻をキーにして降水量データをマージ
hiu_data = pd.merge(hiu_data, precipitation_data, left_on='datetime', right_on='datetime', how='left', suffixes=('', '_1pre'))

# 'kousuiryou_1pre'という列名に変更
hiu_data.rename(columns={'precipitation': 'kousuiryou_1pre'}, inplace=True)

# 変更を保存する
hiu_data.to_csv('data/edited_akashio_data/HIU_data_all_data2.csv', index=False)
