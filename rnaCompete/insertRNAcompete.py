import os
import pandas as pd
from database.databaseMethods import DatabaseMethods
from database.startConfig import StartConfig

def clean_instance_id(instance_id):
    """Clean instance ID to match PDB format"""
    return instance_id.lower().replace(',', '').replace('_', '')

def create_rnacompete_table(database):
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

def insert_rnacompete_data(tsv_file):
    """Process RNAcompete TSV file and insert data into database"""
    # Initialize database connection
    database = DatabaseMethods()
    
    # Create table if it doesn't exist
    create_rnacompete_table(database)
    
    # Read TSV file
    df = pd.read_csv(tsv_file, sep='\t')
    
    # Define columns
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
        values = (
            clean_instance_id(row['instance_id']),  # PDBId
            row['class'],                           # Class
            row['dataset_id'],                      # ProteinId
            row['rbp_name'],                        # ProteinName
            row['organism'],                        # ProteinOrganism
            row['protein_seq'],                     # ProteinSequence
            'A',                                    # ProteinChainIDs
            row['protein_len'],                     # ProteinLength
            1,                                      # NumberProteins
            row['probe_seq'],                       # RNASequence
            'B',                                    # RNAChainIDs
            len(row['probe_seq']),                  # RNALength
            1,                                      # NumberRNAs
            row['mer_seq'],                         # RNAMotif
            len(row['mer_seq']),                    # RNAMotifLength
            row['mer_z'],                           # RNAMotif_z_score
            row['probe_val'],                       # BindingIntensity
            'RNACompete',                           # Experiment
            '',                                     # RNA_GlycosidicBond
            '',                                     # RNA_SugarPucker
            '',                                     # RNA_GammaAngle
            ''                                      # Secondary_Elements
        )
        
        try:
            database.insert_data('exp_protein_rna', columns, values)
            count += 1
        except Exception as e:
            print(f"Error inserting record: {e}")
            continue
    
    # Close database connection
    database.close_connection()
    
    print(f"Successfully inserted {count} records into database")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print(f"Usage: python {os.path.basename(sys.argv[0])} <rnacompete_tsv_file>")
        sys.exit(1)
        
    tsv_file = sys.argv[1]
    if not os.path.exists(tsv_file):
        print(f"Error: File not found: {tsv_file}")
        sys.exit(1)
        
    insert_rnacompete_data(tsv_file)