# ライブラリーの読み込み
import pandas as pd                      #基本ライブラリー
from statsmodels.tsa.seasonal import STL #STL分解
import matplotlib.pyplot as plt          #グラフ描写
# LSTM_real_data\past_data_plot.py
plt.style.use('ggplot')

url="data/edited_akashio_data/HIU_data_+n.csv" #データセットのあるURL
table=pd.read_csv(url,                      #読み込むデータのURL
                  index_col='datetime',        #変数「Month」をインデックスに設定
                  parse_dates=True)         #インデックスを日付型に設定
table.head()
# 特定の列を除外する
columns_to_exclude = ['minute', 'hour']  # 表示させたくない列を指定
table_without_excluded_columns = table.drop(columns=columns_to_exclude)


plt.rcParams['figure.figsize'] = [12, 9]
table_without_excluded_columns.plot()

plt.title('data')                            #グラフタイトル
plt.ylabel('plot') #y

plt.xlabel('datetime')                                #ヨコ軸のラベル
plt.show()