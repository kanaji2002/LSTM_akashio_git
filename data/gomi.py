import pandas as pd

# Read the CSV files
df_total_original = pd.read_csv("edited_akashio_data/total_edited_data.csv", encoding='shift_jis')
df_nissyaryou = pd.read_csv("edited_akashio_data/nissyaryou.csv", encoding='shift_jis')


df_total=pd.DataFrame()
df_total['datetime'] = pd.to_datetime(df_total_original['datetime']).dt.date
df_total['hour']=pd.to_datetime(df_total_original['datetime']).dt.hour.round()
df_total['minute']=pd.to_datetime(df_total_original['datetime']).dt.minute.round()
df_total[[ 'Tem', 'DO', 'Sal','Chl.a']]=df_total_original[[ 'Tem', 'DO', 'Sal', 'Chl.a']]

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
merged_df = merged_df[['datetime', 'hour', 'minute', 'Tem', 'DO', 'Sal', 'nissyaryou', 'Chl.a']]


# Save the result to a new CSV file
merged_df.to_csv("edited_akashio_data/merged_data.csv", index=False, encoding='shift_jis')
# Optional: Display the resulting DataFrame
print(merged_df)
