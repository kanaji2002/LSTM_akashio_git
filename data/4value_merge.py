import pandas as pd

# Read the CSV files
df_total_original = pd.read_csv("data/edited_akashio_data/kai_tem_do_sal_chl.csv", encoding='shift_jis')
df_nissyaryou = pd.read_csv("data/edited_akashio_data/nissyaryou.csv", encoding='shift_jis')


df_total=pd.DataFrame()
df_total['datetime'] = pd.to_datetime(df_total_original['datetime']).dt.date
df_total['hour']=pd.to_datetime(df_total_original['datetime']).dt.hour.round().fillna(0).astype(int)
df_total['minute']=pd.to_datetime(df_total_original['datetime']).dt.minute.round().fillna(0).astype(int)


df_total[[ 'kai','Tem', 'DO', 'Sal','Chl.a']]=df_total_original[[ 'kai','Tem', 'DO', 'Sal', 'Chl.a']]

# df_total['hour'] = df_total['hour'].astype(int)
# df_total['minute'] = df_total['minute'].astype(int)


df_nissyaryou['datetime'] = pd.to_datetime(df_nissyaryou['datetime']).dt.date

# print('df_total')
print(df_total)
print(df_nissyaryou)


# df_total['year'] = df_total['datetime'].dt.year
# df_total['month'] = df_total['datetime'].dt.month
# df_total['day'] = df_total['datetime'].dt.day



# Merge the DataFrames based on the 'datetime' column
merged_df = pd.merge(df_total, df_nissyaryou, how='left', on='datetime')


# # Create a new column 'nissyaryou' in total_edited_data and fill it with values from nissyaryou_edit
merged_df['nissyaryou'] = merged_df['goukei']

#合計はネーミングセンスないからけす．
merged_df = merged_df.drop(columns=['goukei'])
# merged_df = merged_df.drop(columns=['year'])
# merged_df = merged_df.drop(columns=['month'])
# merged_df = merged_df.drop(columns=['day'])

# 列の順序と列名（ラベル）を入れ替える
merged_df = merged_df[['datetime', 'hour', 'minute', 'kai','Tem', 'DO', 'Sal', 'nissyaryou', 'Chl.a']]



# # datetime カラムを文字列に変換してから新しい datetime カラムを作成
# df['new_datetime'] = pd.to_datetime(df['datetime'].astype(str) + ' ' + df['hour'].astype(str) + ':' + df['minute'].astype(str), format='%Y-%m-%d %H:%M')


# merged_df['datetime'] = pd.to_datetime(merged_df['datetime'].astype(str) + ' ' + merged_df['hour'].astype(str) + ':' + merged_df['minute'].astype(str), format='%Y-%m-%d %H:%M')

# merged_df = merged_df.drop(columns=['hour'])
# merged_df = merged_df.drop(columns=['minute'])

# 欠損値を削除
df = merged_df.dropna(subset=['datetime'])

# # Save the result to a new CSV file
merged_df.to_csv("data/edited_akashio_data/k_t_d_s_c_n_merged_data.csv", index=False, encoding='shift_jis')

# Optional: Display the resulting DataFrame
print(merged_df)
