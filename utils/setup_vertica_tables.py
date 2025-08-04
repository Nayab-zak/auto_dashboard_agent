#!/usr/bin/env python3
# setup_vertica_tables.py - Script to create schema, tables and upload data

import os
import sys
import logging
import pandas as pd
from pathlib import Path

# Add parent directory to path for importing
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.vertica_utils import get_vertica_connection, create_schema
from config import VERTICA_TABLE_VOYAGES, VERTICA_TABLE_LINES, VERTICA_TABLE_BOX_STAT

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('setup_vertica')

def extract_table_name(full_table_name):
    """Extract table name from full name with schema"""
    if '.' in full_table_name:
        return full_table_name.split('.')[-1]
    return full_table_name

def get_csv_filename(table_name):
    """Maps table names to CSV filenames in case they don't match exactly"""
    mapping = {
        "Voyages": "VOYAGES",
        "DM_LINES": "DM_LINES",
        "EDW_VOYAGE_BOX_STAT": "EDW_VOYAGE_BOX_STAT"
    }
    
    if table_name in mapping:
        return mapping[table_name]
    return table_name

def extract_schema_name(full_table_name):
    """Extract schema name from full name with schema"""
    if '.' in full_table_name:
        return full_table_name.split('.')[0]
    return None

def create_table_from_csv(conn, csv_path, table_name, schema_name):
    """Create a table in Vertica from a CSV file"""
    try:
        cursor = conn.cursor()
        
        # Read the CSV file to get column names
        logger.info(f"Reading CSV file: {csv_path}")
        df = pd.read_csv(csv_path, low_memory=False)
        
        # Convert all problematic columns to string to avoid data type issues
        logger.info(f"Analyzing column types for {len(df.columns)} columns")
        
        # Create columns definition - use VARCHAR for everything to be safe
        columns = []
        for col in df.columns:
            # Determine the max length for each column
            max_len = df[col].astype(str).str.len().max()
            max_len = max(max_len, 255)  # Minimum 255 for safety
            
            # Use VARCHAR for all columns to avoid type conversion issues
            col_type = f"VARCHAR({max_len})"
            
            # Quote column name to handle special characters
            columns.append(f'"{col}" {col_type}')
            
        columns_str = ", ".join(columns)
        
        # Drop table if exists
        cursor.execute(f"DROP TABLE IF EXISTS {schema_name}.{table_name} CASCADE")
        
        # Create the table
        create_query = f"CREATE TABLE {schema_name}.{table_name} ({columns_str})"
        logger.info(f"Creating table with query: {create_query}")
        cursor.execute(create_query)
        
        # Use COPY command to load data
        abs_path = os.path.abspath(csv_path)
        logger.info(f"Loading data from {abs_path}")
        
        # Use REJECTED DATA and EXCEPTIONS to handle errors
        copy_query = f"""
        COPY {schema_name}.{table_name} FROM LOCAL '{abs_path}' 
        DELIMITER ',' ENCLOSED BY '"' SKIP 1 DIRECT 
        REJECTED DATA '/tmp/{table_name}_rejected.txt'
        EXCEPTIONS '/tmp/{table_name}_exceptions.txt'
        """
        
        cursor.execute(copy_query)
        row_count = cursor.rowcount
        
        logger.info(f"Successfully loaded {row_count} rows into {schema_name}.{table_name}")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Failed to create table from CSV: {e}")
        # Try to provide more information about the error
        if hasattr(e, 'args') and len(e.args) > 0:
            logger.error(f"Error details: {e.args[0]}")
        raise

def main():
    """Main function to create schema, tables and upload data"""
    try:
        # Tables and corresponding CSV files
        tables = [
            VERTICA_TABLE_VOYAGES,
            VERTICA_TABLE_LINES,
            VERTICA_TABLE_BOX_STAT
        ]
        
        # Extract schema name (should be the same for all tables)
        schema_name = extract_schema_name(VERTICA_TABLE_VOYAGES)
        if not schema_name:
            logger.error("Could not extract schema name from table names")
            return False
        
        logger.info(f"Creating schema: {schema_name}")
        
        # Create the schema
        connection = get_vertica_connection()
        cursor = connection.cursor()
        cursor.execute(f"CREATE SCHEMA IF NOT EXISTS {schema_name}")
        cursor.close()
        
        # Find CSV files in temp/data directory
        data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "temp", "data")
        csv_files = {}
        
        for table in tables:
            table_name = extract_table_name(table)
            csv_filename = get_csv_filename(table_name)
            
            logger.info(f"Processing table {table_name}, looking for files with name {csv_filename}")
            
            # Look for exact match
            csv_path = os.path.join(data_dir, f"{csv_filename}.csv")
            if os.path.exists(csv_path):
                logger.info(f"Found CSV file for {table_name}: {csv_path}")
                csv_files[table] = csv_path
                continue
                
            # Look for file containing the table name as last resort
            for file in os.listdir(data_dir):
                if file.endswith(".csv") and (table_name.lower() in file.lower() or csv_filename.lower() in file.lower()):
                    logger.info(f"Found matching CSV file for {table_name}: {file}")
                    csv_files[table] = os.path.join(data_dir, file)
                    break
                        
            if table not in csv_files:
                logger.warning(f"Could not find CSV file for table {table}")
        
        # Create tables and upload data
        for table, csv_path in csv_files.items():
            table_name = extract_table_name(table)
            logger.info(f"Processing table {table_name} with data from {csv_path}")
            create_table_from_csv(connection, csv_path, table_name, schema_name)
            
        connection.close()
        logger.info("All tables created and data uploaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error during setup: {e}")
        return False

if __name__ == "__main__":
    print("Starting setup of Vertica tables...")
    result = main()
    print(f"Setup completed with result: {result}")
