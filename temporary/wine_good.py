import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


#pandas でcsvファイルを読み込む
wine=pd.read_csv("winequality-white2.csv",delimiter=";")

#ワインのグレードを表す列だけ取り出す
y=wine["quality"]

print(y)

'''
#3Dで描画

xname="alcohol"
yname="sulphates"
zname="total sulfur dioxide"

# xname="fixed acidity"
# yname="volatile acidity"
# zname="citric acid"


plt.style.use('ggplot')
fig=plt.figure()
ax=Axes3D(fig)
ax.set_xlabel(xname)
ax.set_ylabel(yname)
ax.set_zlabel(zname)
ax.scatter3D(
    wine[xname],
    wine[yname],
    wine[zname],
    c=y, s=y**2,cmap="cool"
)
plt.show()

'''