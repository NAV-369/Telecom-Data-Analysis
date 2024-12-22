import sqlite3
import pandas as pd
import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_database():
    """Set up SQLite database and import data from CSV files."""
    try:
        # Create data directory if it doesn't exist
        data_dir = Path(__file__).parent.parent / 'data'
        db_path = data_dir / 'telecom.db'
        
        # Create SQLite database
        conn = sqlite3.connect(str(db_path))
        logger.info(f"Created database at {db_path}")
        
        # Read CSV files (assuming they're in the data directory)
        try:
            handsets_df = pd.read_csv(data_dir / 'handsets.csv')
            xdr_df = pd.read_csv(data_dir / 'xdr_data.csv')
            
            # Save to SQLite
            handsets_df.to_sql('handsets', conn, if_exists='replace', index=False)
            xdr_df.to_sql('xdr_data', conn, if_exists='replace', index=False)
            
            logger.info("Successfully imported data to SQLite database")
            
        except FileNotFoundError:
            logger.error("CSV files not found. Please ensure handsets.csv and xdr_data.csv exist in the data directory")
            raise
            
    except Exception as e:
        logger.error(f"Error setting up database: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    setup_database()
