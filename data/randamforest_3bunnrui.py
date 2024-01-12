from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
#import seaborn as sns
import matplotlib.pyplot as plt
# CSVファイルを読み込む
df = pd.read_csv("edited_akashio_data/HIU_data_+n.csv")

# 空白をNaNに変換
df.replace(' ', pd.NA, inplace=True)

# NaNを含む行を削除
df.dropna(inplace=True)

# Chl.aを3つのクラスに分割
df['label_class'] = pd.cut(df['Chl.a'], bins=[-np.inf, 1, np.inf], labels=[0, 1])

# 説明変数として使用する列を選択
selected_columns = ['Tem', 'DO','Sal', 'nissyaryou']
# 正解データの列を指定
label_column = 'label_class'

# データの前処理
labels = df[label_column]
data = df[selected_columns]

# 訓練データとテストデータに分割
data_train, data_test, label_train, label_test = train_test_split(data, labels, test_size=0.1, random_state=0)

# ランダムフォレストのアルゴリズムを利用して学習
clf = RandomForestClassifier(n_estimators=100, random_state=0)
clf.fit(data_train, label_train)

# テストデータで予測
label_pred = clf.predict(data_test)

# 混同マトリックスを取得
conf_matrix = confusion_matrix(label_test, label_pred)

# 分類精度を表示
accuracy = accuracy_score(label_test, label_pred)
print(f"Accuracy: {accuracy:.3f}")

# 分類レポートを表示
report = classification_report(label_test, label_pred)
print("Classification Report:")
print(report)

# 混同マトリックスを表示
print("Confusion Matrix:")
print(conf_matrix)

# 混同マトリックスを可視化
labels = df['label_class'].cat.categories
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()