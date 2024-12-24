import pandas as pd
import numpy as np
import logging
from pathlib import Path
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_table_data(sql_file_path, table_name):
    """Extract data for a specific table from SQL dump file."""
    data_lines = []
    inside_table = False
    
    logger.info(f"Extracting data for table {table_name}...")
    
    with open(sql_file_path, 'r') as f:
        for line in f:
            # Check if we're starting the table data
            if line.startswith(f"COPY public.{table_name}"):
                inside_table = True
                # Get column names from the CREATE TABLE statement
                continue
            
            # Check if we're ending the table data
            if inside_table and line.strip() == '\\.' or line.strip() == ');':
                inside_table = False
                continue
                
            # If we're inside the table data section, add the line
            if inside_table and line.strip():
                data_lines.append(line.strip())
    
    return data_lines

def process_sql_dump():
    """Process the SQL dump file and convert it to CSV."""
    try:
        # Get paths
        data_dir = Path(__file__).parent.parent / 'data'
        sql_file = data_dir / 'telecom.sql'
        
        # Extract XDR data
        logger.info("Processing XDR data...")
        xdr_data_lines = extract_table_data(sql_file, 'xdr_data')
        
        # Convert to DataFrame
        xdr_df = pd.DataFrame([line.split('\t') for line in xdr_data_lines])
        
        # Get column names from SQL file
        with open(sql_file, 'r') as f:
            sql_content = f.read()
            # Extract column names from CREATE TABLE statement
            create_table_match = re.search(r'CREATE TABLE public\.xdr_data \((.*?)\);', 
                                         sql_content, re.DOTALL)
            if create_table_match:
                columns = []
                for line in create_table_match.group(1).split('\n'):
                    if '"' in line:
                        col_name = re.search(r'"([^"]+)"', line)
                        if col_name:
                            columns.append(col_name.group(1))
                
                # Set column names
                xdr_df.columns = columns
                
                # Convert numeric columns
                for col in xdr_df.columns:
                    try:
                        xdr_df[col] = pd.to_numeric(xdr_df[col])
                    except:
                        pass  # Keep as string if conversion fails
                
                # Save to parquet format (more efficient than CSV)
                output_file = data_dir / 'xdr_data.parquet'
                xdr_df.to_parquet(output_file)
                logger.info(f"Data saved to {output_file}")
                
                # Print some basic statistics
                logger.info("\nDataset Statistics:")
                logger.info(f"Number of records: {len(xdr_df)}")
                logger.info(f"Number of columns: {len(xdr_df.columns)}")
                logger.info("\nColumn names:")
                for col in xdr_df.columns:
                    logger.info(f"- {col}")
                
            else:
                raise ValueError("Could not find CREATE TABLE statement in SQL file")
        
    except Exception as e:
        logger.error(f"Error processing SQL dump: {str(e)}")
        raise

if __name__ == "__main__":
    process_sql_dump()
