#!/usr/bin/env python3
"""
Startup script for the dashboard agent system
- Sets up the environment
- Checks database connectivity
- Starts the Flask application
"""
import os
import sys
import subprocess
import argparse
import time
from pathlib import Path

# Add the current directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(__file__))

def create_required_directories():
    """Create the necessary directories if they don't exist"""
    print("Creating required directories...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Required directories
    directories = [
        os.path.join(base_dir, "logs"),
        os.path.join(base_dir, "temp", "data"),
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"  ✓ {directory}")

def check_required_files():
    """Check if all required files are present"""
    print("Checking required files...")
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    required_files = [
        ".env",
        "app.py",
        "config.py",
        "table_selector_agent_prod.py",
        "kpi_planner_agent_prod.py",
        "sql_generator_agent_prod.py",
        "dashboard_creator_agent_cte.py",
        "agent1_schema_metadata_with_samples.json"
    ]
    
    all_files_present = True
    for file in required_files:
        file_path = os.path.join(base_dir, file)
        if not os.path.exists(file_path):
            print(f"  ✗ Missing required file: {file}")
            all_files_present = False
        else:
            print(f"  ✓ {file}")
    
    if not all_files_present:
        print("Error: Some required files are missing!")
        sys.exit(1)

def check_env_variables():
    """Check if all required environment variables are set"""
    print("Checking environment variables...")
    
    # Import config after we've confirmed it exists
    from config import (OPENAI_API_KEY, VERTICA_HOST, VERTICA_PORT, VERTICA_DB, 
                      VERTICA_USER, VERTICA_PASSWORD, VERTICA_TABLE_VOYAGES, 
                      VERTICA_TABLE_LINES, VERTICA_TABLE_BOX_STAT)
    
    required_vars = {
        "OPENAI_API_KEY": OPENAI_API_KEY,
        "VERTICA_HOST": VERTICA_HOST,
        "VERTICA_PORT": VERTICA_PORT,
        "VERTICA_DB": VERTICA_DB,
        "VERTICA_USER": VERTICA_USER,
        "VERTICA_PASSWORD": VERTICA_PASSWORD,
        "VERTICA_TABLE_VOYAGES": VERTICA_TABLE_VOYAGES,
        "VERTICA_TABLE_LINES": VERTICA_TABLE_LINES,
        "VERTICA_TABLE_BOX_STAT": VERTICA_TABLE_BOX_STAT
    }
    
    all_vars_present = True
    for var_name, var_value in required_vars.items():
        if not var_value:
            print(f"  ✗ Missing environment variable: {var_name}")
            all_vars_present = False
        else:
            masked_value = var_value[:3] + "..." + var_value[-3:] if var_name == "OPENAI_API_KEY" else var_value
            print(f"  ✓ {var_name}: {masked_value}")
    
    if not all_vars_present:
        print("Error: Some required environment variables are missing!")
        sys.exit(1)

def check_database_connectivity():
    """Check if we can connect to the Vertica database"""
    print("Checking database connectivity...")
    try:
        from utils.db_checker import get_connection
        conn = get_connection()
        if conn:
            conn.close()
            print("  ✓ Successfully connected to Vertica database")
            return True
        else:
            print("  ✗ Failed to connect to Vertica database")
            return False
    except Exception as e:
        print(f"  ✗ Error checking database connection: {e}")
        return False

def check_dependencies():
    """Check if all required Python packages are installed"""
    print("Checking Python dependencies...")
    
    required_packages = [
        "flask", "flask-cors", "pandas", "numpy", "openai", 
        "streamlit", "pyodbc", "plotly", "python-dotenv"
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - Not installed")
            missing_packages.append(package)
    
    if missing_packages:
        print("\nSome required packages are missing. Install them with:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def setup_database(args):
    """Set up the database tables if needed"""
    if args.skip_db_setup:
        print("Skipping database setup as requested...")
        return True
    
    print("Setting up database tables...")
    try:
        from utils.vertica_setup import drop_tables, create_table_from_csv, load_csv_to_table
        from config import VERTICA_TABLE_VOYAGES, VERTICA_TABLE_LINES, VERTICA_TABLE_BOX_STAT
        
        # Check if data files exist
        base_dir = os.path.dirname(os.path.abspath(__file__))
        data_dir = os.path.join(base_dir, "temp", "data")
        
        voyages_csv = os.path.join(data_dir, "VOYAGES.csv")
        lines_csv = os.path.join(data_dir, "DM_LINES.csv")
        box_stat_csv = os.path.join(data_dir, "EDW_VOYAGE_BOX_STAT.csv")
        
        if not os.path.exists(voyages_csv) or not os.path.exists(lines_csv):
            print("Warning: CSV data files not found in temp/data directory.")
            print("Please add the required CSV files before proceeding.")
            return False
        
        # Only proceed with setup if force flag is set or user confirms
        if not args.force_db_setup:
            confirmation = input("Do you want to proceed with database setup? This will drop existing tables. (y/n): ")
            if confirmation.lower() != 'y':
                print("Database setup skipped.")
                return True
        
        # Drop tables if they exist
        drop_tables()
        
        # Create tables from CSV headers
        if os.path.exists(voyages_csv):
            create_table_from_csv(voyages_csv, VERTICA_TABLE_VOYAGES)
        if os.path.exists(lines_csv):
            create_table_from_csv(lines_csv, VERTICA_TABLE_LINES)
        if os.path.exists(box_stat_csv):
            create_table_from_csv(box_stat_csv, VERTICA_TABLE_BOX_STAT)
        
        # Load data
        if os.path.exists(voyages_csv):
            load_csv_to_table(voyages_csv, VERTICA_TABLE_VOYAGES)
        if os.path.exists(lines_csv):
            load_csv_to_table(lines_csv, VERTICA_TABLE_LINES)
        if os.path.exists(box_stat_csv):
            load_csv_to_table(box_stat_csv, VERTICA_TABLE_BOX_STAT)
        
        print("Database setup completed successfully.")
        return True
        
    except Exception as e:
        print(f"Error setting up database: {e}")
        return False

def start_app(args):
    """Start the Flask application"""
    print("Starting Flask application...")
    
    # Get the app.py path
    app_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app.py")
    
    # Use the specified host and port or defaults
    host = args.host or "0.0.0.0"
    port = args.port or 5000
    
    # Configure command
    cmd = [
        sys.executable,
        app_path
    ]
    
    # Set environment variables for Flask
    env = os.environ.copy()
    env["FLASK_APP"] = app_path
    env["FLASK_ENV"] = "development" if args.debug else "production"
    
    print(f"Launching app on {host}:{port} (debug={args.debug})...")
    
    try:
        # For non-background mode, just exec the process
        if not args.background:
            os.execve(sys.executable, cmd, env)
        # For background mode, use subprocess
        else:
            logs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
            log_file = os.path.join(logs_dir, "app.log")
            
            with open(log_file, "a") as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env
                )
            
            print(f"App started in background with PID {process.pid}")
            print(f"Log file: {log_file}")
            
            # Write PID to file for later management
            with open(os.path.join(logs_dir, "app.pid"), "w") as f:
                f.write(str(process.pid))
            
    except Exception as e:
        print(f"Error starting app: {e}")
        return False
    
    return True

def run_diagnostics():
    """Run database diagnostics"""
    print("Running database diagnostics...")
    try:
        from utils.db_checker import run_diagnostics
        run_diagnostics()
        return True
    except Exception as e:
        print(f"Error running diagnostics: {e}")
        return False

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Dashboard Agent System startup script")
    
    parser.add_argument("--skip-checks", action="store_true", help="Skip environment and dependency checks")
    parser.add_argument("--skip-db-setup", action="store_true", help="Skip database setup")
    parser.add_argument("--force-db-setup", action="store_true", help="Force database setup without confirmation")
    parser.add_argument("--diagnostics", action="store_true", help="Run database diagnostics and exit")
    parser.add_argument("--host", help="Host to bind the Flask app to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, help="Port to run the Flask app on (default: 5000)")
    parser.add_argument("--debug", action="store_true", help="Run Flask in debug mode")
    parser.add_argument("--background", action="store_true", help="Run the app in the background")
    
    return parser.parse_args()

def main():
    """Main function"""
    args = parse_args()
    
    print("\n=== Dashboard Agent System Setup ===\n")
    
    # Create required directories
    create_required_directories()
    
    # Check required files
    check_required_files()
    
    # Run diagnostics if requested and exit
    if args.diagnostics:
        run_diagnostics()
        return
    
    # Skip remaining checks if requested
    if not args.skip_checks:
        # Check environment variables
        check_env_variables()
        
        # Check Python dependencies
        if not check_dependencies():
            print("\nPlease install the missing dependencies before continuing.")
            return
        
        # Check database connectivity
        if not check_database_connectivity():
            print("\nPlease check your database configuration before continuing.")
            return
    
    # Setup database if needed
    if not setup_database(args):
        print("\nDatabase setup incomplete or had errors. Check logs for details.")
        if not args.force_db_setup:
            confirmation = input("Continue anyway? (y/n): ")
            if confirmation.lower() != 'y':
                return
    
    # Start the Flask app
    print("\nAll checks passed. Starting application...\n")
    start_app(args)

if __name__ == "__main__":
    main()
