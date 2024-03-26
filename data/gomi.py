import pandas as pd
import numpy as np

dfs=[]
#CSVファイルの読み込み

file1 = f'data/edited_akashio_data/HIU_data_all_data.csv'
file2 = f'kousuiryou/kousuiryou.csv'
file3 = f'tyouryuu/get_uv_250m_csv/merge.csv'
df1 = pd.read_csv(file1, encoding='shift_jis')
df2 = pd.read_csv(file2, encoding='shift_jis')
df_t = pd.read_csv(file3, encoding='shift_jis')

# 日付と時間を結合してdatetime列を作成
df1['datetime'] = pd.to_datetime(df1['datetime'])
df2['datetime'] = pd.to_datetime(df2['datetime'])
df_t['datetime'] = pd.to_datetime(df_t['datetime'])

# datetime列を使って結合する
merged_df = pd.merge(df1, df2, on='datetime')

# df1のdatetime列を日付型に変換
merged_df['datetime'] = pd.to_datetime(df1['datetime'])

# 日付と時間を分割してdatetime列を作成
df_t[['date', 'hour', 'time']] = df_t['datetime'].str.split(' ', expand=True)
df_t['hour'] = df_t['hour'].str.split(':').str[0]  # 時刻の部分を取り除く
df_t['hour'] = df_t['hour'].astype(int)  # hour列のデータ型を整数型に変換
df_t['datetime'] = pd.to_datetime(df_t['date'] + ' ' + df_t['hour'].astype(str) + ':00')


# 結合
merged_df = pd.merge(merged_df, df_t, on=['datetime', 'hour'])

# 結合したデータを表示
print(merged_df)
