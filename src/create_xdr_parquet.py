import pandas as pd
import numpy as np

# Load the original data (assuming it is in CSV format)
# Replace 'path_to_your_xdr_data.csv' with the actual path to your XDR data CSV
csv_file_path = '../data/xdr_data.csv'
parquet_file_path = '../data/xdr_data.parquet'

def create_parquet():
    # Load the CSV data
    df = pd.read_csv(csv_file_path)

    # Perform necessary data cleaning and transformations
    # Example: Replace '\N' with NaN and convert to numeric
    df.replace('\N', np.nan, inplace=True)
    df = df.apply(pd.to_numeric, errors='ignore')  # Adjust as needed for your data

    # Save the DataFrame to Parquet format
    df.to_parquet(parquet_file_path, index=False)
    print(f'Created {parquet_file_path}')  

if __name__ == '__main__':
    create_parquet()
