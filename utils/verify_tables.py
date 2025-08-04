#!/usr/bin/env python3
# verify_tables.py - Script to verify the created tables

import sys
import os
import logging

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vertica_utils import get_vertica_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('verify_tables')

def main():
    """Main function to verify tables in DPW_DL schema"""
    try:
        connection = get_vertica_connection()
        cursor = connection.cursor()
        
        # Check all tables in the schema
        logger.info("Checking tables in DPW_DL schema:")
        cursor.execute("""
            SELECT table_name
            FROM v_catalog.tables 
            WHERE table_schema = 'DPW_DL'
            ORDER BY table_name
        """)
        tables = cursor.fetchall()
        
        if not tables:
            logger.warning("No tables found in DPW_DL schema")
            return False
        
        logger.info(f"Found {len(tables)} tables in DPW_DL schema:")
        for table_name in tables:
            logger.info(f"- {table_name[0]}")
        
        # Check row counts for each table
        for table_row in tables:
            table_name = table_row[0]
            cursor.execute(f"SELECT COUNT(*) FROM DPW_DL.{table_name}")
            count = cursor.fetchone()[0]
            logger.info(f"Table DPW_DL.{table_name} has {count} rows")
        
        cursor.close()
        connection.close()
        return True
    except Exception as e:
        logger.error(f"Error during verification: {e}")
        return False

if __name__ == "__main__":
    print("Verifying tables in DPW_DL schema...")
    result = main()
    print(f"Verification completed with result: {result}")
