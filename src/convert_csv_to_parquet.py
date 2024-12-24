import pandas as pd
import logging

# Upgrade: Added error handling and logging
# Upgrade: Optimized CSV to Parquet conversion using chunking
logging.basicConfig(level=logging.INFO)

# Load the CSV file
csv_file_path = '../data/handsets.csv'
parquet_file_path = '../data/handsets.parquet'

chunk_size = 10000  # Number of rows per chunk

def convert_csv_to_parquet():
    try:
        # Create an empty DataFrame to hold the chunks
        df_list = []
        # Read the CSV file in chunks
        for chunk in pd.read_csv(csv_file_path, chunksize=chunk_size):
            df_list.append(chunk)
        # Concatenate all chunks into a single DataFrame
        full_df = pd.concat(df_list)
        # Convert and save as Parquet
        full_df.to_parquet(parquet_file_path, index=False)
        logging.info(f'Converted {csv_file_path} to {parquet_file_path}')
    except Exception as e:
        logging.error(f'Error converting file: {e}')

if __name__ == '__main__':
    convert_csv_to_parquet()
