#!/usr/bin/env python3
# delete_dpw_dl_schema.py - Script to delete the DPW_DL schema

import os
import logging
import sys
from db_operations import delete_schema

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger('delete_schema')

def main():
    """Main function to delete DPW_DL schema."""
    schema_name = "DPW_DL"
    logger.info(f"Starting deletion of schema: {schema_name}")
    
    try:
        success = delete_schema(schema_name)
        if success:
            logger.info(f"Schema {schema_name} has been successfully deleted")
        else:
            logger.warning(f"Schema {schema_name} was not deleted (might not exist)")
    except Exception as e:
        logger.error(f"Error during schema deletion: {e}")
        return False
    
    return True

if __name__ == "__main__":
    main()
