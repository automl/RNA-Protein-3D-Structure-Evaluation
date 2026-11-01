import os
import glob
from Bio.PDB import MMCIFParser
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from database.startConfig import StartConfig
from database.databaseMethods import DatabaseMethods
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize database connection
config = StartConfig()
db = DatabaseMethods()

def clean_experimental_details(details):
    """Clean experimental details text by removing quotes and semicolons."""
    if not details or details == '?' or details.lower() == 'null':
        return None
        
    # Remove leading/trailing whitespace
    details = details.strip()
    
    # Remove semicolons at the beginning and end
    if details.startswith(';'):
        details = details[1:]
    if details.endswith(';'):
        details = details[:-1]
    
    # Remove single quotes at the beginning and end
    if details.startswith("'") and details.endswith("'"):
        details = details[1:-1]
    
    # Remove double quotes at the beginning and end
    if details.startswith('"') and details.endswith('"'):
        details = details[1:-1]
    
    # Clean up any remaining whitespace
    details = details.strip()
    
    return details if details else None

def get_sample_concentration(pdb_info):
    """Extract sample concentration from NMR data."""
    try:
        concentration = pdb_info.get('_pdbx_nmr_exptl_sample.concentration', ['?'])[0]
        units = pdb_info.get('_pdbx_nmr_exptl_sample.concentration_units', ['?'])[0]
        
        if concentration != '?' and concentration.lower() != 'null' and units != '?' and units.lower() != 'null':
            return f"{concentration} {units}"
        elif concentration != '?' and concentration.lower() != 'null':
            return concentration
        else:
            return None
    except:
        return None

def add_exp_columns():
    """Add exp_Solution and exp_Sample_Conc columns if they don't exist."""
    try:
        exp_table = config.get_reference_table_name()
        
        # Check exp_protein_rna columns
        columns_query = f"PRAGMA table_info({exp_table})"
        columns = db.execute_query(columns_query)
        column_names = [col[1] for col in columns]
        
        if 'exp_Solution' not in column_names:
            db.execute_query(f"ALTER TABLE {exp_table} ADD COLUMN exp_Solution TEXT")
            logger.info(f"Added exp_Solution column to {exp_table}")
            
        # if 'exp_Sample_Conc' not in column_names:
        #     db.execute_query(f"ALTER TABLE {exp_table} ADD COLUMN exp_Sample_Conc TEXT")
        #     logger.info(f"Added exp_Sample_Conc column to {exp_table}")
            
    except Exception as e:
        logger.error(f"Error adding exp columns: {str(e)}")
        return False
    return True

def process_structure(pdb_id, cif_file):
    """Process a single structure and extract experimental solution and concentration."""
    try:
        pdb_info = MMCIF2Dict(cif_file)
        exp_solution = None
        exp_sample_conc = None
        title = None
        
        # First check the experimental method
        method = pdb_info.get('_exptl.method', ['?'])[0]
        logger.info(f"Processing {pdb_id} with method: {method}")
        
        if method == 'X-RAY DIFFRACTION':
            # Get experimental solution from crystal growth conditions
            details = pdb_info.get('_exptl_crystal_grow.pdbx_details', ['?'])[0]
            exp_solution = clean_experimental_details(details)
            title = pdb_info.get('_citation.title', ['?'])[0]
                
        elif method == 'ELECTRON MICROSCOPY':
            # Get experimental solution from EM buffer details
            details = pdb_info.get('_em_buffer.details', ['?'])[0]
            exp_solution = clean_experimental_details(details)
            title = pdb_info.get('_citation.title', ['?'])[0]
            # if exp_solution == None:
            #     exp_solution = pdb_info.get('_citation.title', ['?'])[0]
            #     exp_solution = f'# {exp_solution}'
                    
        elif method == 'SOLUTION NMR':
            # Get experimental solution from NMR sample conditions
            details = pdb_info.get('_pdbx_nmr_exptl_sample_conditions.details', ['?'])[0]
            exp_solution = clean_experimental_details(details)
            title = pdb_info.get('_citation.title', ['?'])[0]
            
            # Get sample concentration for NMR
            exp_sample_conc = get_sample_concentration(pdb_info)
        else:
            logger.warning(f"Unknown experimental method for {pdb_id}: {method}")
                    
        return exp_solution, exp_sample_conc, title
        
    except Exception as e:
        logger.error(f"Error processing {pdb_id}: {str(e)}")
        return None, None

def update_database():
    """Update database with experimental solution and concentration from CIF files."""
    if not add_exp_columns():
        return
        
    # Get reference folder path
    ref_folder = config.ref_folder
    exp_table = config.get_reference_table_name()
    
    # Get all structures from database
    query = f"SELECT PDBId FROM {exp_table}"
    structures = db.execute_query(query)
    
    updated_count = 0
    for (pdb_id,) in structures:
        if not pdb_id:
            continue
            
        # Find CIF file
        cif_file = os.path.join(ref_folder, f"{pdb_id.lower()}.cif")
        if not os.path.exists(cif_file):
            logger.warning(f"CIF file not found for {pdb_id}")
            continue
            
        # Process structure
        exp_solution, exp_sample_conc, title = process_structure(pdb_id, cif_file)
        
        # Update database
        try:
            if exp_sample_conc == None:
                query = f"""
                                UPDATE {exp_table} 
                                SET exp_Solution = ?, Comment = ?
                                WHERE PDBId = ?
                                """
                db.execute_query(query, (exp_solution, title, pdb_id))
            else:
                query = f"""
                UPDATE {exp_table} 
                SET exp_Solution = ?, exp_Sample_Conc = ?, Comment = ?
                WHERE PDBId = ?
                """
                db.execute_query(query, (exp_solution, exp_sample_conc, title, pdb_id))
            
            if exp_solution: # or exp_sample_conc:
                solution_preview = f"exp_Solution = {exp_solution[:100]}{'...' if exp_solution and len(exp_solution) > 100 else ''}" if exp_solution else "exp_Solution = None"
                # conc_preview = f"exp_Sample_Conc = {exp_sample_conc}" if exp_sample_conc else "exp_Sample_Conc = None"
                logger.info(f"Updated {pdb_id}: {solution_preview}")
                updated_count += 1
            else:
                logger.info(f"No experimental solution or concentration found for {pdb_id}")
                
        except Exception as e:
            logger.error(f"Error updating database for {pdb_id}: {str(e)}")
    
    logger.info(f"Successfully updated {updated_count} structures with experimental solution and concentration")

if __name__ == "__main__":
    update_database() 