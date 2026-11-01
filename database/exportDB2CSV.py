import sqlite3
import csv
import os
import sys
from database.databaseMethods import DatabaseMethods

database = DatabaseMethods()

def convert_table(table, out_dir=None):
    """Write <table>.csv to out_dir (mainExecution.py passes --folder), default <repo>/results."""
    conn = sqlite3.connect(database.db_path)
    cursor = conn.cursor()
    cursor.execute(f'SELECT * FROM {table}')
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_path = out_dir or os.path.join(project_root, "results")
    os.makedirs(results_path, exist_ok=True) # Make directory if it doesn't exist
    save_path = os.path.join(results_path, f"{table}.csv")

    rows = cursor.fetchall()
    with open(save_path, 'w', newline='', encoding='utf-8') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([i[0] for i in cursor.description])

        for row in rows:
            formatted_row = [
                f"{item:.8f}" if isinstance(item, float) else item
                for item in row
            ]
            csv_writer.writerow(formatted_row)

    conn.close()

def convert_all_dbs(source_folder, table_name):
    project_root = os.path.abspath(source_folder)
    results_path = os.path.join(project_root, "results")
    os.makedirs(results_path, exist_ok=True)

    for filename in os.listdir(source_folder):
        if filename.endswith('.db'):
            db_path = os.path.join(source_folder, filename)
            csv_name = os.path.splitext(filename)[0] + '.csv'
            save_path = os.path.join(results_path, csv_name)

            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            try:
                cursor.execute(f"SELECT * FROM {table_name}")
                rows = cursor.fetchall()
                print(save_path)
                with open(save_path, 'w', newline='', encoding='utf-8') as file:
                    csv_writer = csv.writer(file)
                    csv_writer.writerow([i[0] for i in cursor.description])
                    for row in rows:
                        formatted_row = [
                            f"{item:.2f}" if isinstance(item, float) else item
                            for item in row
                        ]
                        csv_writer.writerow(formatted_row)

                print(f"Saved: {csv_name}")
            except sqlite3.Error as e:
                print(f"Failed to process {filename}: {e}")
            finally:
                conn.close()

# convert_all_dbs("data/9May2025_config/synthetic80-81/combined_DB", "pred_protein_rna")

if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(f"Usage: python {os.path.basename(sys.argv[0])} <table> [output folder; default results/]")
        sys.exit(1)
    table = sys.argv[1]
    convert_table(table, sys.argv[2] if len(sys.argv) == 3 else None)
    print(f"Exported {table} -> {os.path.join(sys.argv[2] if len(sys.argv) == 3 else 'results', table + '.csv')}")
    # print(f'Data exported to {csv_file}')