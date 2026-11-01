import os
import pandas as pd
from database.databaseMethods import DatabaseMethods

def create_protein_rna_table(database):
    """Create exp_protein_rna table if it doesn't exist"""
    query = """
    CREATE TABLE IF NOT EXISTS exp_protein_rna (
        PDBId TEXT PRIMARY KEY,
        Class TEXT,
        ProteinId TEXT,
        ProteinName TEXT,
        ProteinOrganism TEXT,
        ProteinSequence TEXT,
        ProteinChainIDs TEXT,
        ProteinLength INTEGER,
        NumberProteins INTEGER,
        RNASequence TEXT,
        RNAChainIDs TEXT,
        RNALength INTEGER,
        NumberRNAs INTEGER,
        RNAMotif TEXT,
        RNAMotifLength INTEGER,
        RNAMotif_z_score FLOAT,
        BindingIntensity FLOAT,
        Experiment TEXT,
        RNA_GlycosidicBond TEXT,
        RNA_SugarPucker TEXT,
        RNA_GammaAngle TEXT,
        Secondary_Elements TEXT
    )
    """
    database.execute_query(query)

def insert_protein_rna_data(csv_file):
    """Process exported_data.csv file and insert data into database"""
    # Initialize database connection
    database = DatabaseMethods()
    
    # Create table if it doesn't exist
    create_protein_rna_table(database)
    
    # Read CSV file
    df = pd.read_csv(csv_file, sep=';')
    
    # Define columns for the database table
    columns = [
        'PDBId', 'Class', 'ProteinId', 'ProteinName', 'ProteinOrganism',
        'ProteinSequence', 'ProteinChainIDs', 'ProteinLength', 'NumberProteins',
        'RNASequence', 'RNAChainIDs', 'RNALength', 'NumberRNAs',
        'RNAMotif', 'RNAMotifLength', 'RNAMotif_z_score', 'BindingIntensity',
        'Experiment', 'RNA_GlycosidicBond', 'RNA_SugarPucker', 'RNA_GammaAngle',
        'Secondary_Elements'
    ]
    
    # Process each row
    count = 0
    for _, row in df.iterrows():
        # Calculate RNAMotifLength from RNAMotif
        rna_motif_length = len(str(row['RNAMotif'])) if pd.notna(row['RNAMotif']) else 0
        
        # Determine Class based on RNAMotif value
        rna_class = "minprobe" if str(row['RNAMotif']) == "0" else "maxprobe"
        
        values = (
            row['PDBId'],                # PDBId
            rna_class,                   # Class
            '',                          # ProteinId
            row['ProteinName'],          # ProteinName
            row['Organism'],             # ProteinOrganism
            row['ProteinSequence'],      # ProteinSequence
            '',                          # ProteinChainIDs
            row['ProteinLength'],        # ProteinLength
            1,                           # NumberProteins
            row['RNASequence'],          # RNASequence
            '',                          # RNAChainIDs
            row['RNALength'],            # RNALength
            1,                           # NumberRNAs
            row['RNAMotif'],             # RNAMotif
            rna_motif_length,            # RNAMotifLength
            None,                        # RNAMotif_z_score
            None,                        # BindingIntensity
            row['Experiment'],           # Experiment
            '',                          # RNA_GlycosidicBond
            '',                          # RNA_SugarPucker
            '',                          # RNA_GammaAngle
            ''                           # Secondary_Elements
        )
        
        try:
            database.insert_data('exp_protein_rna', columns, values)
            count += 1
        except Exception as e:
            print(f"Error inserting record for PDB ID {row['PDBId']}: {e}")
            continue
    
    # Close database connection
    database.close_connection()
    
    print(f"Successfully inserted {count} records into database")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} exported_data.csv")
        sys.exit(1)
        
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"Error: File not found: {csv_file}")
        sys.exit(1)
        
    insert_protein_rna_data(csv_file)