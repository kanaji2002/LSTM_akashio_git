from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv("edited_akashio_data/HIU_data_+n.csv")
df = df.replace('-', pd.NA,).dropna()
# データの前処理
# labelには正解データが，dataにはその特徴量が入っている．
labels = df['Chl.a']
data = df.drop([ 'datetime','minute','kai'], axis=1)

# 学習データとテストデータに分割
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

# ランダムフォレストのアルゴリズムを利用して学習
clf = RandomForestRegressor(criterion="absolute_error", n_estimators=100, oob_score=True)
oob_errors = []

# 学習過程での誤差を計算
for i in range(1, 101):
    clf.set_params(n_estimators=i)
    clf.fit(data_train, label_train)
    oob_errors.append(1 - clf.oob_score_)

# 誤差率のプロット
plt.plot(range(1, 101), oob_errors, label='OOB Error Rate')
plt.xlabel('Number of Trees')
plt.ylabel('Error Rate')
plt.legend()
plt.show()