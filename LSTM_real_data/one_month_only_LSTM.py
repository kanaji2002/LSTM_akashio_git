# ライブラリーの読み込み
import pandas as pd                      #基本ライブラリー
from statsmodels.tsa.seasonal import STL #STL分解
import matplotlib.pyplot as plt          #グラフ描写
# one_month_only_LSTM
plt.style.use('ggplot')

url="data/edited_akashio_data/HIU_data_+n.csv" #データセットのあるURL
table=pd.read_csv(url,                      #読み込むデータのURL
                  index_col='datetime',        #変数「Month」をインデックスに設定
                  parse_dates=True)         #インデックスを日付型に設定
table.head()

plt.rcParams['figure.figsize'] = [12, 9]

table.plot()
plt.title('data')                            #グラフタイトル
plt.ylabel('plot') #y

plt.xlabel('datetime')                                #ヨコ軸のラベル
plt.show()
stl=STL(table['Passengers'], period=12, robust=True)
stl_series = stl.fit()
stl_series.plot()
plt.show()