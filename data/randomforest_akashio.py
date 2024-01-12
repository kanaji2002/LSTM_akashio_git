from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
def yosoku():
    # CSVファイルを読み込む
    df = pd.read_csv("edited_akashio_data/HIU_data_+n.csv")
    df = df.replace('-', pd.NA).dropna()

    # 説明変数として使用する列を選択
    selected_columns = ['Tem', 'DO', 'Sal', 'nissyaryou']
    # 正解データの列を指定
    label_column = 'Chl.a'

    # データの前処理
    labels = df[label_column]
    data = df[selected_columns]

    results = []
    for i in range(1, 2):
        mean_errors = []
        for j in range(0, 10):
            np.random.seed(j)
            # 訓練データとテストデータに分割
            data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

            # ランダムフォレストのアルゴリズムを利用して学習
            clf = RandomForestRegressor(criterion="absolute_error", n_estimators=100)
            clf.fit(data_train, label_train)

            # テストデータで予測
            label_pred = clf.predict(data_test)

            # 予測結果と誤差率を表示
            for true_label, prediction in zip(label_test, label_pred):
                error = abs(true_label - prediction) / prediction * 100
                print(f"True Label: {true_label:.3f}, Predicted Label: {prediction:.3f}, Error Rate: {error:.3f}%")
            
            # 平均誤差率を計算
            mean_errors.append(np.mean([abs(true_label - prediction) / prediction * 100 for true_label, prediction in zip(label_test, label_pred)]))

    # 各Seedでの平均誤差率を表示
    for k, mean_error in enumerate(mean_errors):
        print(f"Seed {k+1:2}: {mean_error:.3f}%")

    # 平均平均誤差率を計算
    mean_mean_error = np.mean(mean_errors)
    results.append(mean_mean_error)

    # 平均平均誤差率を表示
    [print(f"平均平均誤差率 {i:.3f}%") for i in results]