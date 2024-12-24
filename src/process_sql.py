import subprocess
import logging
from pathlib import Path
import os

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_postgresql():
    """Set up PostgreSQL database and import data."""
    try:
        # Create a new database
        db_name = "telecom_analysis"
        
        # Drop database if it exists
        subprocess.run(['dropdb', '--if-exists', db_name], check=True)
        
        # Create new database
        subprocess.run(['createdb', db_name], check=True)
        logger.info(f"Created database: {db_name}")
        
        # Get path to SQL file
        sql_file = Path(__file__).parent.parent / 'data' / 'telecom.sql'
        
        # Import data
        with open(os.devnull, 'w') as devnull:
            subprocess.run(['psql', '-d', db_name, '-f', str(sql_file)], 
                         check=True, stdout=devnull)
        logger.info("Successfully imported data")
        
        return db_name
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error setting up database: {str(e)}")
        raise

if __name__ == "__main__":
    setup_postgresql()
