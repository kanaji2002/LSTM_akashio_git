import json

# 提供されたJSONデータ
data = '''
{
    "raspberry_cpu_id": "100000009560d53d",
    "temperature": {
        "125cm": 21.9375,
        "100cm": 22.0,
        "75cm": 22.0625,
        "50cm": 21.9375,
        "25cm": 22.0625
    },
    "oxygen": {
        "timestamp": "2023-11-02 09:15:06",
        "sensors": {
            "97": {
                "name": "do-sensor",
                "moduletype": "DO",
                "data": "8.04"
            }
        }
    }
}
'''

# JSONデータをPython辞書にパース
data_dict = json.loads(data)

# 温度情報を取得
temperature_data = data_dict["temperature"]

# 酸素濃度情報を取得
oxygen_data = data_dict["oxygen"]["sensors"]["97"]["data"]

# 結果を表示
print("温度データ:", temperature_data)
print("酸素濃度データ:", oxygen_data,"mg/L")