import pandas as pd

# Read the CSV files
df_total_original = pd.read_csv("data/edited_akashio_data/total_edited_data.csv", encoding='shift_jis')
df_nissyaryou = pd.read_csv("data/edited_akashio_data/nissyaryou.csv", encoding='shift_jis')

df_total = pd.DataFrame()
df_total['datetime'] = pd.to_datetime(df_total_original['datetime']).dt.date
df_total['hour'] = pd.to_datetime(df_total_original['datetime']).dt.hour.round().fillna(0).astype(int)
df_total['minute'] = pd.to_datetime(df_total_original['datetime']).dt.minute.round().fillna(0).astype(int)
df_total[['kai','Tem', 'DO', 'Sal', 'Chl.a']] = df_total_original[['kai','Tem', 'DO', 'Sal', 'Chl.a']]

df_nissyaryou['datetime'] = pd.to_datetime(df_nissyaryou['datetime']).dt.date

# Merge the DataFrames based on the 'datetime' column
merged_df = pd.merge(df_total, df_nissyaryou, how='left', on='datetime')

# Create a new column 'nissyaryou' in total_edited_data and fill it with values from nissyaryou_edit
merged_df['nissyaryou'] = merged_df['goukei']

# # Drop unnecessary columns
# merged_df = merged_df.drop(columns=['goukei', 'hour', 'minute'])

# Convert 'datetime', 'hour', and 'minute' to a new 'datetime' column
merged_df['datetime'] = pd.to_datetime(merged_df['datetime'].astype(str) + ' ' + merged_df['hour'].astype(str) + ':' + merged_df['minute'].astype(str), format='%Y-%m-%d %H:%M')

# # Drop 'hour' and 'minute' columns
# merged_df = merged_df.drop(columns=['hour', 'minute'])

# Drop rows with missing values
merged_df = merged_df.dropna(subset=['datetime'])

# Save the result to a new CSV file
merged_df.to_csv("data/edited_akashio_data/merged_data.csv", index=False, encoding='shift_jis')

# Optional: Display the resulting DataFrame
print(merged_df)
