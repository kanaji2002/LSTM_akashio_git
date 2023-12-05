import pandas as pd
import numpy as np

dfs=[]
#CSVファイルの読み込み


for recorded_year in range(1982,2021):
    filename=f'akashio-data/{recorded_year}.csv'
    df = pd.read_csv(filename,encoding='shift_jis')
    # df = df.apply(lambda x: x.strip() if isinstance(x, str) else x)
    
    # "day"列から小数点が含まれる行を削除（日付で小数点のところがあったから削除）
    df = df[~df['day'].astype(str).str.contains('\.')]

    #dataframeの特殊文字だけを置き換える．
    df['Chl.a']=df['Chl.a'].str.replace('>','')
    df['Chl.a']=df['Chl.a'].str.replace('<','')
    
    
    # "day"列からハイフンが含まれる行を削除#どちらでもいける．
    #df = df[~df['day'].astype(str).str.contains('-')]
    df.drop(df[df['day'] == '-'].index, inplace=True)
    
    
    # 'year', 'mon', 'day', 'time' 列を結合して日時の列を作成
    df['datetime'] = pd.to_datetime(df['year'].astype(str) + '-' + df['mon'].astype(str) + '-' + df['day'].astype(str) + ' ' + df['time'], format='%Y-%m-%d %H:%M', errors='coerce')

    # 日時でソート
    df = df.sort_values(by='datetime')

    # 不要な列を削除（必要に応じて変更）
    df = df[['datetime', 'Tem', 'DO', 'Sal', 'Chl.a']]

   
    # # 'Chl.a' 列における "-" を含む行を削除
    df.drop(df[df['Tem'] == '-'].index, inplace=True)
    df.drop(df[df['DO'] == '-'].index, inplace=True)
    df.drop(df[df['Sal'] == '-'].index, inplace=True)
    df.drop(df[df['Chl.a'] == '-'].index, inplace=True)


    
    dfs.append(df)
    

    # 上層データ（U)と，下層データ（B)があるが，赤潮は基本上層に現れ，クロロフィル濃度をみても上層の方が値が出やすいので，上層だけを保存．
    # ソートされたデータを新しいCSVファイルに保存
    # save_filename=f'edited_akashio_data/{recorded_year}.csv'
    # df.to_csv(save_filename, index=False,header=True,encoding='shift_jis')
merged = pd.concat(dfs, axis=0) 
merged.to_csv('edited_akashio_data/total_edited_data.csv', index=False,header=True,encoding='shift_jis')

