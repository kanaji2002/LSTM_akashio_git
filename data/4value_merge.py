import pandas as pd

# Read the CSV files
df_total_original = pd.read_csv("edited_akashio_data/HIU_data.csv", encoding='shift_jis')
df_nissyaryou = pd.read_csv("edited_akashio_data/nissyaryou.csv", encoding='shift_jis')


df_total=pd.DataFrame()
df_total['datetime'] = pd.to_datetime(df_total_original['datetime']).dt.date
df_total['hour']=pd.to_datetime(df_total_original['datetime']).dt.hour.round().fillna(0).astype(int)
df_total['minute']=pd.to_datetime(df_total_original['datetime']).dt.minute.round().fillna(0).astype(int)


df_total[[ 'kai','Tem', 'DO', 'Sal','Chl.a']]=df_total_original[[ 'kai','Tem', 'DO', 'Sal', 'Chl.a']]

df_nissyaryou['datetime'] = pd.to_datetime(df_nissyaryou['datetime']).dt.date

# print('df_total')
print(df_total)
print(df_nissyaryou)

# Merge the DataFrames based on the 'datetime' column
merged_df = pd.merge(df_total, df_nissyaryou, how='left', on='datetime')


# # Create a new column 'nissyaryou' in total_edited_data and fill it with values from nissyaryou_edit
merged_df['nissyaryou'] = merged_df['goukei']

#合計はネーミングセンスないからけす．
merged_df = merged_df.drop(columns=['goukei'])

# 列の順序と列名（ラベル）を入れ替える
merged_df = merged_df[['kai','datetime', 'hour', 'minute','Tem', 'DO', 'Sal', 'nissyaryou', 'Chl.a']]

# 欠損値を削除
df = merged_df.dropna(subset=['datetime'])

# # Save the result to a new CSV file
merged_df.to_csv("edited_akashio_data/HIU_data_+n.csv", index=False, encoding='shift_jis')

# Optional: Display the resulting DataFrame
print(merged_df)
