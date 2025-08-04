#!/usr/bin/env python3
# db_operations.py - Utility for database operations

import sys
import os
import vertica_python
import logging

# Add parent directory to path for importing config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import VERTICA_HOST, VERTICA_PORT, VERTICA_DB, VERTICA_USER, VERTICA_PASSWORD

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('db_operations')

def get_vertica_connection():
    """Create and return a connection to Vertica database."""
    try:
        conn_info = {
            'host': VERTICA_HOST,
            'port': int(VERTICA_PORT),
            'database': VERTICA_DB,
            'user': VERTICA_USER,
            'password': VERTICA_PASSWORD,
            'autocommit': True
        }
        
        logger.info(f"Connecting to Vertica at {VERTICA_HOST}:{VERTICA_PORT}")
        connection = vertica_python.connect(**conn_info)
        logger.info("Successfully connected to Vertica")
        return connection
    except Exception as e:
        logger.error(f"Failed to connect to Vertica: {e}")
        raise

def delete_schema(schema_name):
    """Delete the specified schema from Vertica."""
    try:
        connection = get_vertica_connection()
        cursor = connection.cursor()
        
        # Check if schema exists
        cursor.execute(f"SELECT schema_name FROM v_catalog.schemata WHERE schema_name = '{schema_name}'")
        if cursor.fetchone() is None:
            logger.warning(f"Schema '{schema_name}' does not exist")
            return False
        
        logger.info(f"Attempting to drop schema '{schema_name}'")
        
        # Drop the schema with CASCADE option to remove all dependent objects
        cursor.execute(f"DROP SCHEMA {schema_name} CASCADE")
        logger.info(f"Successfully dropped schema '{schema_name}'")
        
        # Close cursor and connection
        cursor.close()
        connection.close()
        logger.info("Connection closed")
        return True
        
    except Exception as e:
        logger.error(f"Failed to drop schema '{schema_name}': {e}")
        raise

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python db_operations.py <schema_name>")
        sys.exit(1)
    
    schema_name = sys.argv[1]
    try:
        delete_schema(schema_name)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
