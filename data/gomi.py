import pandas as pd

# Read the CSV files
df_total = pd.read_csv("edited_akashio_data/total_edited_data.csv", encoding='shift_jis')
df_nissyaryou = pd.read_csv("edited_akashio_data/nissyaryou.csv", encoding='shift_jis')

# Convert the 'datetime' column to datetime format for both DataFrames
df_total['datetime'] = pd.to_datetime(df_total['datetime']).dt.date
df_nissyaryou['datetime'] = pd.to_datetime(df_nissyaryou['datetime']).dt.date  # 日付型に変換

# Merge the DataFrames based on the 'datetime' column
merged_df = pd.merge(df_total, df_nissyaryou, how='left', on='datetime')

# Extract hour and minute from the merged 'datetime' column
merged_df['hour'] = pd.to_datetime(merged_df['datetime']).dt.hour
merged_df['minute'] = pd.to_datetime(merged_df['datetime']).dt.minute

# Create a new column 'nissyaryou' in total_edited_data and fill it with values from nissyaryou_edit
merged_df['nissyaryou'] = merged_df['goukei']

# Remove unnecessary columns
merged_df = merged_df.drop(columns=['goukei'])

# Reorder columns
merged_df = merged_df[['datetime', 'hour', 'minute', 'Tem', 'DO', 'Sal', 'nissyaryou', 'Chl.a']]

# Save the result to a new CSV file
merged_df.to_csv("edited_akashio_data/merged_data.csv", index=False, encoding='shift_jis')

# Optional: Display the resulting DataFrame
print(merged_df)
