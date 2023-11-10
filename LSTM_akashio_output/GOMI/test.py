import pandas as pd
# データフレームの作成
df = pd.DataFrame({'A': [1, 2, 3, 4, 5],
                   'B': ['apple', 'banana', 'orange', 'apple', 'grape']})
# 特定の行を削除する
print(df)
df.drop([4], inplace=True)

print(df)
from YORIGOMI import kannsuu
a=kannsuu.kaesu()
print(a)