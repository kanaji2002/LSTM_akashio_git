import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path

# filepath='winequality-white.csv'

# filepath='C:/Users/Kanaji Rinntarou/Desktop/kennkyuu/LSTM_akashio/LSTM_akashio_output/number.csv'

filepath='LSTM_akashio_output/number.csv'
# # # Load the wine data from a CSV file
# wine = pd.read_csv("winequality-white.csv")

# print(Path(filepath).read_text())
wine2 = pd.read_csv(filepath)
print(wine2)

import os

# カレントディレクトリのパスを取得
current_directory = os.getcwd()

# カレントディレクトリを表示
print("カレントディレクトリ:", current_directory)
