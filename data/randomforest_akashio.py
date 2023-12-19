from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# CSVファイルを読み込む
df = pd.read_csv("data/edited_akashio_data/merged_data_not_nan copy.csv")

# データの前処理
# labelには正解データが，dataにはその特徴量が入っている．
labels = df['Chl.a']
data = df.drop(['hour', 'minute','kai'], axis=1)


# data["team_exp"] = -2 ** data["team_exp"]
# data["term"] = data["term"] ** (1 / 2)


results = []
for i in range(1, 2):
    mean_errors = []
    for j in range(0, 10):
        np.random.seed(j)
        # 訓練データとテストデータに分割
        # test_sizeで，テストに使う割合を決めることができる．
        data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

        # ランダムフォレストのアルゴリズムを利用して学習
        clf = RandomForestRegressor(criterion="absolute_error")
        clf.fit(data_train, label_train)

        # テストデータで予測
        label_pred = clf.predict(data_test)

        errors = []
        # テストデータでの予測結果を表示
        for true_label, prediction in zip(label_test, label_pred):
            # 「soutaigosa」は，各テストデータに対して使用する
            error = abs(true_label - prediction) / prediction * 100
            errors.append(error)
            # print(f"True Label: {true_label}, Predicted Label: {prediction}, SoutaiGosa: {error}%")
        mean_errors.append(np.mean(errors))
        print(f"Seed{j:2}: {np.mean(errors):.3}%")
    results.append(np.mean(mean_errors))

[print(f"平均平均誤差率 {i:.3}%") for i in results]

# plt.plot(results)
# plt.ylim(0, 100)
# plt.show()
