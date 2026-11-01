import sqlite3
import os
import argparse
from pathlib import Path
from database.startConfig import StartConfig


def merge_tables(source_db_path, merged_db_path, table_name):
    """
    Copy a table from source database into merged database.
    
    Args:
        source_db_path (str): Path to the source database
        merged_db_path (str): Path to the merged database
        table_name (str): Name of the table to merge
    """
    source_conn = sqlite3.connect(source_db_path)
    merged_conn = sqlite3.connect(merged_db_path)
    
    try:
        # Get schema and create table if it doesn't exist
        schema_query = f"SELECT sql FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        schema_result = source_conn.execute(schema_query).fetchone()
        if not schema_result:
            print(f"Table {table_name} not found in {os.path.basename(source_db_path)}")
            return
        schema = schema_result[0]
        
        table_exists = merged_conn.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'").fetchone() is not None
        if not table_exists:
            merged_conn.execute(schema)
        else:
            # Table exists - check if source database has additional columns
            # Get columns from source database
            source_columns = [description[0] for description in source_conn.execute(f"SELECT * FROM {table_name} LIMIT 1").description]
            # Get columns from merged database
            merged_columns = [description[0] for description in merged_conn.execute(f"SELECT * FROM {table_name} LIMIT 1").description]
            # Find missing columns
            missing_columns = [col for col in source_columns if col not in merged_columns]
            # Add missing columns to merged table
            for col in missing_columns:
                # Get column type from source database schema
                pragma_query = f"PRAGMA table_info({table_name})"
                source_pragma = source_conn.execute(pragma_query).fetchall()
                for col_info in source_pragma:
                    if col_info[1] == col:  # col_info[1] is column name
                        col_type = col_info[2]  # col_info[2] is column type
                        try:
                            merged_conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {col} {col_type}")
                            merged_conn.commit()
                            print(f"Added missing column '{col}' to merged table {table_name}")
                        except sqlite3.OperationalError as e:
                            if "duplicate column" not in str(e).lower():
                                raise
            
        # Determine the key column for duplicate checking (not necessarily the primary key)
        # Use PDBId for exp tables, exp_db_id for pred tables
        config = StartConfig()
        logical_key_column = "PDBId" if table_name == config.ref_table else "exp_db_id"
        
        # Get source data
        source_data = source_conn.execute(f"SELECT * FROM {table_name}").fetchall()
        if not source_data:
            print(f"No data found in {os.path.basename(source_db_path)} for table {table_name}")
            return
            
        source_columns = [description[0] for description in source_conn.execute(f"SELECT * FROM {table_name}").description]
        
        # Check for Id column - it's an SQLite internal primary key
        # If it exists, we need to exclude it when inserting data
        id_column_index = None
        columns_to_use = []
        for i, col in enumerate(source_columns):
            if col.lower() == 'id':
                id_column_index = i
            else:
                columns_to_use.append(col)
                
        columns_str = ', '.join(columns_to_use)
        
        # Find the index of our logical key column
        logical_key_index = source_columns.index(logical_key_column)
        
        # Check which values already exist in the merged database
        existing_values = set()
        if table_exists:
            existing_query = f"SELECT {logical_key_column} FROM {table_name}"
            existing_values = set(row[0] for row in merged_conn.execute(existing_query).fetchall())
        
        # Filter out rows with duplicate logical keys
        new_rows = []
        duplicate_rows = []
        for row in source_data:
            if row[logical_key_index] in existing_values:
                duplicate_rows.append(row)
            else:
                # Only include columns that aren't the Id
                if id_column_index is not None:
                    new_row = [v for i, v in enumerate(row) if i != id_column_index]
                    new_rows.append(new_row)
                else:
                    new_rows.append(row)
        
        # Insert only new rows
        if new_rows:
            placeholders = ', '.join(['?' for _ in columns_to_use])
            insert_query = f"INSERT INTO {table_name} ({columns_str}) VALUES ({placeholders})"
            merged_conn.executemany(insert_query, new_rows)
            merged_conn.commit()
        
        print(f"Added {len(new_rows)} new rows from {os.path.basename(source_db_path)} " +
              f"(skipped {len(duplicate_rows)} existing rows out of {len(source_data)} total)")
        
    except Exception as e:
        print(f"Error processing {os.path.basename(source_db_path)}: {str(e)}")
        raise
    finally:
        source_conn.close()
        merged_conn.close()


def merge_databases(folder_path):
    """
    Create a new merged database from two source databases in the folder.
    
    Args:
        folder_path (str): Path to folder containing the databases
    """
    # Find SQLite databases in the folder and sort alphabetically
    # This ensures databases with more columns (e.g., a2021.db) are processed first
    db_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.db') and f != 'merged.db'])
    
    if len(db_files) != 2:
        raise ValueError(f"Expected exactly 2 database files in {folder_path}, found {len(db_files)}")
    
    db1_path = os.path.join(folder_path, db_files[0])
    db2_path = os.path.join(folder_path, db_files[1])
    merged_path = os.path.join(folder_path, 'merged.db')
    
    # Remove merged.db if it exists
    if os.path.exists(merged_path):
        os.remove(merged_path)
    
    # Get table names from StartConfig
    config = StartConfig()
    pred_table = config.pred_table
    exp_table = config.ref_table
    tables = [pred_table, exp_table]
    
    print(f"\nCreating merged database from:")
    print(f"- {db_files[0]}")
    print(f"- {db_files[1]}")
    
    for table in tables:
        print(f"\nProcessing table: {table}")
        merge_tables(db1_path, merged_path, table)
        merge_tables(db2_path, merged_path, table)


def main():
    parser = argparse.ArgumentParser(description='Merge two SQLite3 databases')
    parser.add_argument('folder', type=str, help='Folder containing exactly two .db files')
    
    args = parser.parse_args()
    
    try:
        merge_databases(args.folder)
        print("\nMerge completed successfully! Created 'merged.db'")
    except Exception as e:
        print(f"\nError during merge: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main() 