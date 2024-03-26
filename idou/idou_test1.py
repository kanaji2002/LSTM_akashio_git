import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
def move_cells(a,i,j, u, v, r):
    rows, cols = a.shape

    new_a = np.copy(a)  # 元の配列を変更せずにコピーする
    #u_sum=0 今後実装予定　潮流の合計値を保存，if分で，u_sumを超えたら隣のセルに移動．
    #割り残を使用し，u_sumは，-255~255いないにする．
    for _ in range(3):
        for i in range(rows):
            for j in range(cols):
                if abs(u[i, j]) <= 250 and abs(v[i, j]) <= 250:
                    continue  # uとvの値が250以下の場合、移動しない

                # # 描画する円の範囲を計算
                # min_i = max(0, i - int(r * 1.5))
                # max_i = min(rows, i + int(r * 1.5) + 1)
                # min_j = max(0, j - int(r * 1.5))
                # max_j = min(cols, j + int(r * 1.5) + 1)

                # for x in range(min_i, max_i):
                #     for y in range(min_j, max_j):
                #         if (x - i) ** 2 + (y - j) ** 2 <= r ** 2:
                #             new_a[x, y] = 1  # 赤く描画

                # 移動条件に基づいて新しい位置を決定
                #右上を基準に時計周り
                if 250 <= u and 250 <= v: #1
                    new_i = i+1
                    new_j = j+1
                elif 250 <= u and -255 <= v < 250:#2
                    new_i = i+1
                    new_j = j
                elif 250 <= u  and v < -255: #3
                    new_i = i+1
                    new_j = j-1
                elif -255 <= u < 250 and v < -255:#4
                    new_i = i
                    new_j = j-1
                elif u < -255 and  v < -255:#5
                    new_i = i-1
                    new_j = j-1
                elif u < -255 and  -255 <= v < 250:#6
                    new_i = i-1
                    new_j = j
                elif u < -255 and  250 <= v:#7
                    new_i = i-1
                    new_j = j+1
                elif -255 <= u < 250 and  250 <= v:#8
                    new_i = i-1
                    new_j = j+1
                
                # 2.	Axesオブジェクト生成
                fig, ax = plt.subplots(figsize=(4,4))
                
                ax.set_xticks([-2, -1, 0, 1, 2])
                ax.set_yticks([-2, -1, 0, 1, 2])
                ax.grid()
                # # 移動先が描画範囲内であれば、移動する
                # if min_i <= new_i < max_i and min_j <= new_j < max_j:
                #     new_a[new_i, new_j] = 2  # 移動先を別の色で描画
                c = patches.Circle( xy=(new_i,new_j), radius=r*1.5)# 円のオブジェクト
                ax.add_patch(c)
    return new_a

# テスト用のデータを作成
a = np.zeros((300, 300))
u = np.random.uniform(-500, 500, (300, 300))
v = np.random.uniform(-500, 500, (300, 300))
i=20
j=30
# テスト用の関数を呼び出して、新しい配列を取得
new_a = move_cells(a, i,j,u, v, 3)

# 結果を表示
plt.imshow(new_a, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.show()
