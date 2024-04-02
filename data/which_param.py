import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from boruta import BorutaPy

# CSVファイルを読み込む
data = pd.read_csv('data/edited_akashio_data/HIU_data_all_data2.csv')  # ファイル名は適切に変更してください

# 特徴量と目的変数を分割
X = data.drop(columns=['Chl.a','kai','datetime'])  # 目的変数を含まない特徴量
y = data['Chl.a']  # 目的変数

# RandomForestRegressorでBorutaを実行
rf = RandomForestRegressor(n_jobs=-1, max_depth=5)
feat_selector = BorutaPy(rf, n_estimators='auto', verbose=2, random_state=1)

feat_selector.fit(X.values, y.values)  # int型への変換を削除

# 選択された特徴量を確認
selected = feat_selector.support_
print('選択された特徴量の数: %d' % np.sum(selected))
print(selected)
print(X.columns[selected])

# 選択した特徴量で学習
X_selected = X[X.columns[selected]]
rf2 = RandomForestRegressor(n_jobs=-1, max_depth=5)
rf2.fit(X_selected, y)
print('SCORE with selected Features: %1.2f' % rf2.score(X_selected, y))
# 特徴量のランキングを取得
feature_ranks = feat_selector.ranking_

# 特徴量の名前を取得
feature_names = X.columns

# ランキングと特徴量名を組み合わせて表示
ranked_features = list(zip(feature_names, feature_ranks))
# 重要度の順にソートして表示
ranked_features_sorted = sorted(ranked_features, key=lambda x: x[1])

# 結果の表示
print("特徴量のランキング:")
for feature, rank in ranked_features_sorted:
    print(f"{feature}: {rank}")
