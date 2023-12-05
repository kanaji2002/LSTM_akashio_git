import pandas as pd
import numpy as np

dfs=[]
#CSVファイルの読み込み

filename=f'akashio-data/nissyaryou.csv'
df = pd.read_csv(filename,encoding='shift_jis')
# df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)
df[['year','mon','day']]=df['datetime'].str.split('/',expand=True)

df['datetime'] = pd.to_datetime(df['year'] + '-' + df['mon'] + '-' + df['day'], format='%Y-%m-%d', errors='coerce')

df = df[['datetime', 'goukei']]
    


# 上層データ（U)と，下層データ（B)があるが，赤潮は基本上層に現れ，クロロフィル濃度をみても上層の方が値が出やすいので，上層だけを保存．
# ソートされたデータを新しいCSVファイルに保存
save_filename=f'edited_akashio_data/nissyaryou.csv'
df.to_csv(save_filename, index=False,header=True,encoding='shift_jis')
merged = pd.concat(dfs, axis=0) 
merged.to_csv('edited_akashio_data/total_edited_data.csv', index=False,header=True,encoding='shift_jis')

