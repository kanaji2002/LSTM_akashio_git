import pandas as pd

# CSVファイルのパス
filepath = 'LSTM_akashio_output/CSV/number.csv'

# CSVファイルからデータを読み込み
data = pd.read_csv(filepath, delimiter=';')

# 特定の行を削除（4~7行目）
data = data.drop(data.columns[4:11],axis=1)

# # 前半部分と後半部分を結合
# df_combined = pd.concat([data.iloc[5:], data.iloc[:11]])

df_combined=data
# 新しいCSVファイルに保存
df_combined.to_csv('LSTM_akashio_output/CSV/new_data.csv', index=False, sep=';')
