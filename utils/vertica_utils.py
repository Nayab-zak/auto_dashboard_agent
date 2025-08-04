#!/usr/bin/env python3
# vertica_utils.py - Comprehensive utilities for Vertica database operations

import sys
import os
import vertica_python
import logging
import pandas as pd
import csv

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
logger = logging.getLogger('vertica_utils')

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

def execute_query(query, fetch=True):
    """Execute a SQL query and optionally return results."""
    try:
        connection = get_vertica_connection()
        cursor = connection.cursor()
        
        logger.info(f"Executing query: {query[:100]}...")
        cursor.execute(query)
        
        result = None
        if fetch:
            result = cursor.fetchall()
            column_names = [desc[0] for desc in cursor.description]
            logger.info(f"Query returned {len(result)} rows")
        
        cursor.close()
        connection.close()
        logger.info("Connection closed")
        
        if fetch and result is not None:
            return result, column_names
        return True
    except Exception as e:
        logger.error(f"Failed to execute query: {e}")
        raise

def create_schema(schema_name):
    """Create a new schema in Vertica."""
    try:
        query = f"CREATE SCHEMA IF NOT EXISTS {schema_name}"
        execute_query(query, fetch=False)
        logger.info(f"Successfully created schema '{schema_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to create schema '{schema_name}': {e}")
        raise

def delete_schema(schema_name):
    """Delete a schema from Vertica."""
    try:
        # Check if schema exists
        exists_query = f"SELECT schema_name FROM v_catalog.schemata WHERE schema_name = '{schema_name}'"
        result, _ = execute_query(exists_query)
        
        if not result:
            logger.warning(f"Schema '{schema_name}' does not exist")
            return False
        
        logger.info(f"Attempting to drop schema '{schema_name}'")
        query = f"DROP SCHEMA {schema_name} CASCADE"
        execute_query(query, fetch=False)
        logger.info(f"Successfully dropped schema '{schema_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to drop schema '{schema_name}': {e}")
        raise

def list_schemas():
    """List all schemas in the database."""
    try:
        query = "SELECT schema_name, schema_owner FROM v_catalog.schemata ORDER BY schema_name"
        result, column_names = execute_query(query)
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(result, columns=column_names)
        logger.info(f"Found {len(df)} schemas")
        return df
    except Exception as e:
        logger.error(f"Failed to list schemas: {e}")
        raise

def list_tables(schema_name=None):
    """List all tables in the database or in a specific schema."""
    try:
        if schema_name:
            query = f"""
            SELECT table_schema, table_name, owner_name
            FROM v_catalog.tables 
            WHERE table_schema = '{schema_name}'
            ORDER BY table_schema, table_name
            """
        else:
            query = """
            SELECT table_schema, table_name, owner_name
            FROM v_catalog.tables 
            ORDER BY table_schema, table_name
            """
            
        result, column_names = execute_query(query)
        
        # Convert to DataFrame for better display
        df = pd.DataFrame(result, columns=column_names)
        logger.info(f"Found {len(df)} tables")
        return df
    except Exception as e:
        logger.error(f"Failed to list tables: {e}")
        raise

def create_table_from_csv(csv_path, table_name, schema_name, delimiter=','):
    """Create a table in Vertica from a CSV file."""
    try:
        # Read CSV header to determine columns
        with open(csv_path, 'r') as f:
            reader = csv.reader(f, delimiter=delimiter)
            header = next(reader)
            
            # Sample the first row to guess data types
            sample = next(reader)
            
        # Create a simple schema based on the CSV header
        columns = []
        for i, col_name in enumerate(header):
            # Try to guess the data type based on the sample
            sample_value = sample[i]
            
            # Simple type inference
            if sample_value.replace('.', '', 1).isdigit():
                if '.' in sample_value:
                    col_type = "FLOAT"
                else:
                    col_type = "INT"
            else:
                col_type = "VARCHAR(1000)"
                
            columns.append(f'"{col_name}" {col_type}')
            
        columns_str = ", ".join(columns)
        
        # Create the table
        create_query = f"CREATE TABLE IF NOT EXISTS {schema_name}.{table_name} ({columns_str})"
        execute_query(create_query, fetch=False)
        logger.info(f"Created table {schema_name}.{table_name}")
        
        # Use COPY command to load data
        connection = get_vertica_connection()
        cursor = connection.cursor()
        
        copy_query = f"""
        COPY {schema_name}.{table_name} FROM LOCAL '{os.path.abspath(csv_path)}' 
        DELIMITER '{delimiter}' SKIP 1 DIRECT
        """
        
        cursor.execute(copy_query)
        row_count = cursor.rowcount
        
        cursor.close()
        connection.close()
        
        logger.info(f"Successfully loaded {row_count} rows into {schema_name}.{table_name}")
        return True
    except Exception as e:
        logger.error(f"Failed to create table from CSV: {e}")
        raise

def query_to_dataframe(query):
    """Execute a query and return the results as a pandas DataFrame."""
    try:
        result, column_names = execute_query(query)
        df = pd.DataFrame(result, columns=column_names)
        return df
    except Exception as e:
        logger.error(f"Failed to query data: {e}")
        raise

if __name__ == "__main__":
    print("Vertica Utils - Available functions:")
    print("1. List schemas")
    print("2. List tables in a schema")
    print("3. Create schema")
    print("4. Delete schema")
    print("5. Create table from CSV")
    
    choice = input("Enter your choice (1-5): ")
    
    if choice == "1":
        df = list_schemas()
        print(df)
    elif choice == "2":
        schema = input("Enter schema name (leave empty for all schemas): ")
        df = list_tables(schema if schema else None)
        print(df)
    elif choice == "3":
        schema = input("Enter schema name to create: ")
        create_schema(schema)
    elif choice == "4":
        schema = input("Enter schema name to delete: ")
        delete_schema(schema)
    elif choice == "5":
        csv_path = input("Enter path to CSV file: ")
        table_name = input("Enter table name: ")
        schema_name = input("Enter schema name: ")
        create_table_from_csv(csv_path, table_name, schema_name)
    else:
        print("Invalid choice")
