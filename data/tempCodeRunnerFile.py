
# merged_df['datetime'] = pd.to_datetime(merged_df['datetime'].astype(str) + ' ' + merged_df['hour'].astype(str) + ':' + merged_df['minute'].astype(str), format='%Y-%m-%d %H:%M')

# merged_df = merged_df.drop(columns=['hour'])
# merged_df = merged_df.drop(columns=['minute'])