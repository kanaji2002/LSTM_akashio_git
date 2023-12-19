import pandas as pd

def validate_data(file_path):
    # Read CSV file
    data = pd.read_csv(file_path)

    # Define the column names for numeric variables
    numeric_columns = ['Tem', 'DO', 'Sal', 'nissyaryou', 'Chl.a']

    # Validate each numeric variable
    for col in numeric_columns:
        invalid_rows = data[~((data[col] >= 0) & (data[col] < 100))]
        if not invalid_rows.empty:
            print(f"Error in {col} column:")
            print(invalid_rows)
    
    print("Data validation complete.")

# Replace 'your_file_path.csv' with the actual path to your CSV file
file_path = "C:/Users/Kanaji Rinntarou/OneDrive - 独立行政法人 国立高等専門学校機構/Desktop/kennkyuu/LSTM_akashio/LSTM_real_data/merged_data.csv"
validate_data(file_path)