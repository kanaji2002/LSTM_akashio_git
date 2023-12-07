import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.metrics import mean_squared_error


# データの読み込み
df = pd.read_csv("../data/edited_akashio_data/merged_data.csv")

# 'datetime' カラムを日時型に変換
df['datetime'] = pd.to_datetime(df['datetime'])

# 'datetime' をインデックスに設定
df.set_index('datetime', inplace=True)

# Chl.a 以外の特徴量を選択
features = ['hour', 'minute', 'Tem', 'DO', 'Sal', 'nissyaryou']
X = df[features].values

# Chl.a をターゲットとする
y = df['Chl.a'].values


scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)
y_scaled = scaler.fit_transform(y.reshape(-1, 1))


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, shuffle=False)


model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=2)


y_pred = model.predict(X_test)
y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_test_inv, y_pred_inv)
print(f'Mean Squared Error: {mse}')
