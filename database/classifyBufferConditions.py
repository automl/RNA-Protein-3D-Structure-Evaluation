#!/usr/bin/env python3
"""
Parse exp_Solution strings from database and classify:
- ion concentration category and estimated total salt molarity (simple sum of explicit mol values)
- PEG concentration category and PEG MW
- buffer concentration category
Outputs CSV.
"""

import re
import csv
import pandas as pd
from pathlib import Path
from database.startConfig import StartConfig
from database.databaseMethods import DatabaseMethods
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Initialize database connection
config = StartConfig()
db = DatabaseMethods()

# --- helper patterns
M_PATTERN = re.compile(r'([0-9]*\.?[0-9]+)\s*(M|mM)\b', flags=re.I)
PERCENT_PATTERN = re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%\b')
# PEG_MW_PATTERN = re.compile(r'PEG[-_\s]?([0-9]{2,5})(?:\b|[^0-9])', flags=re.I)
# PEG_K_PATTERN = re.compile(r'PEG[-_\s]?([0-9]{1,2})\s*[Kk]\b')   # PEG 4K style
# PEG_GENERIC = re.compile(r'PEG\b', flags=re.I)

# PEG with explicit MW (e.g. PEG 3350, Polyethylene glycol 4,000)
PEG_MW_PATTERN = re.compile(
    r'(?:PEG|PEMME|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)'
    r'[-_\s]?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]{2,5})(?=[^\d]|$)',
    flags=re.I
)

# PEG with K suffix (e.g. PEG 4K, Polyethylene glycol 20K)
PEG_K_PATTERN = re.compile(
    r'(?:PEG|polyethylene\s+glycol)[-_\s]?([0-9]{1,2})\s*[Kk](?=\b|[^0-9])',
    flags=re.I
)

# Generic PEG (no MW specified)
PEG_GENERIC = re.compile(r'(?:PEG|polyethylene\s+glycol)\b', flags=re.I)

# Define salts for ionic strength calculation
# ---------- SALTS ----------
# Format: canonical name : (cation charge, anion charge)

SALTS = {
    # Chlorides
    'sodium chloride': (1, -1),
    'potassium chloride': (1, -1),
    'lithium chloride': (1, -1),
    'magnesium chloride': (2, -1),
    'manganese chloride': (2, -1),
    'calcium chloride': (2, -1),

    # Sulfates
    'ammonium sulfate': (1, -2),
    'lithium sulfate': (1, -2),
    'sodium sulfate': (1, -2),
    'potassium sulfate': (1, -2),

    # Acetates
    'sodium acetate': (1, -1),
    'potassium acetate': (1, -1),
    'ammonium acetate': (1, -1),
    'magnesium acetate': (2, -1),
    'calcium acetate': (2, -1),

    # Formates
    'sodium formate': (1, -1),
    'magnesium formate': (2, -1),

    # Citrates (citrate³⁻)
    'sodium citrate': (1, -3),
    'trisodium citrate': (1, -3),
    'ammonium citrate': (1, -3),

    # Tartrates (tartrate²⁻)
    'ammonium tartrate': (1, -2),
    'potassium sodium tartrate': (1, -2),

    # Phosphates (averaged –2)
    'sodium phosphate': (1, -2),
    'potassium phosphate': (1, -2),
    'ammonium phosphate': (1, -2),

    # Carbonates (carbonate²⁻)
    'sodium carbonate': (1, -2),
    'potassium carbonate': (1, -2),
    'ammonium carbonate': (1, -2),

    # Succinate (succinate²⁻)
    'disodium succinate': (1, -2),
    'sodium succinate': (1, -2),

    # Halides
    'potassium bromide': (1, -1),
    'sodium bromide': (1, -1),
    'potassium iodide': (1, -1),
    'sodium iodide': (1, -1),
    'ammonium fluoride': (1, -1)
}


# ---------- ALIASES ----------
ALIASES = {
    # Chlorides
    'nacl': 'sodium chloride', 'na cl': 'sodium chloride',
    'kcl': 'potassium chloride', 'k cl': 'potassium chloride',
    'licl': 'lithium chloride', 'li cl': 'lithium chloride',
    'mgcl2': 'magnesium chloride', 'mg cl2': 'magnesium chloride',
    'mncl2': 'manganese chloride', 'mn cl2': 'manganese chloride',
    'cacl2': 'calcium chloride', 'ca cl2': 'calcium chloride',

    # Sulfates
    '(nh4)2so4': 'ammonium sulfate', 'nh4 sulfate': 'ammonium sulfate',
    'ammonium sulfate': 'ammonium sulfate',  # Direct match
    'liso4': 'lithium sulfate', 'li2so4': 'lithium sulfate', 'lis04': 'lithium sulfate',
    'na2so4': 'sodium sulfate',
    'k2so4': 'potassium sulfate',

    # Acetates
    'na acetate': 'sodium acetate', 'naacetate': 'sodium acetate',
    'k acetate': 'potassium acetate',
    'ammonium acetate': 'ammonium acetate',
    'magnesium acetate tetrahydrate': 'magnesium acetate',

    # Formates
    'na formate': 'sodium formate', 'mg formate': 'magnesium formate',
    'mg-formate': 'magnesium formate',

    # Citrates
    'na citrate': 'sodium citrate', 'tri-sodium citrate': 'trisodium citrate',
    'citrate': 'sodium citrate',

    # Tartrates
    'ammonium tartrate': 'ammonium tartrate',
    'nh4/tartrate': 'ammonium tartrate', 'nh4 tartrate': 'ammonium tartrate',
    'potassium sodium tartrate': 'potasium sodium tartrate',  # Map correct spelling to typo key
    'potasium sodium tartrate': 'potasium sodium tartrate',
    'sodium potassium tartrate': 'potasium sodium tartrate',

    # Phosphates (averaged to dibasic)
    'sodium phosphate': 'sodium phosphate',
    'na h2po4': 'sodium phosphate',
    'na2 hpo4': 'sodium phosphate',
    'k phosphate': 'potassium phosphate',

    # Carbonates
    'na2co3': 'sodium carbonate', 'k2co3': 'potassium carbonate',

    # Succinate
    'na2 succinate': 'disodium succinate',

    # Halides
    'kbr': 'potassium bromide', 'nabr': 'sodium bromide',
    'ki': 'potassium iodide', 'nai': 'sodium iodide',
    'naf': 'ammonium fluoride'
}


BUFFERS = [
    'tris', 'tris-hcl', 'tris(hydroxymethyl)aminomethane',
    'hepes', 'mes', 'bicine', 'imidazole',
    'cacodylate', 'cacodyl', 'sodium cacodylate', 'sodium cacodylate trihydrate',
    'bis-tris', 'bis-tris propane', 'bis-trispropane',
    'citric', 'citrate',
    'phosphate', 'sodium phosphate', 'potassium phosphate',
    'acetate', 'formate', 'tartrate',
    'pH', 'mib', 'ches', 'ada', 'succinate'
]

# Hydrate recogniser
HYDRATE_PATTERN = re.compile(r'\b(mono|di|tri|tetra|penta|hexa)hydrate\b', flags=re.I)

def normalize_name(raw):
    """Normalize ingredient names to canonical salt names."""
    r = raw.lower().strip()
    
    # Remove hydrate information (e.g., "tetrahydrate", "monohydrate")
    r = HYDRATE_PATTERN.sub('', r).strip()
    
    # Remove pH information (e.g., "pH 7.4", "ph 7.4")
    r = re.sub(r'\bph\s*\d+\.?\d*\b', '', r, flags=re.I).strip()
    
    # Remove extra whitespace
    r = re.sub(r'\s+', ' ', r).strip()
    
    # simple alias lookup by substring
    for key, val in ALIASES.items():
        if key in r:
            return val
    # direct match
    for salt in SALTS.keys():
        if salt in r:
            return salt
    return None

def parse_concentration(text):
    """Parse concentration: "200 mM", "0.1 M", "0.2M", etc."""
    text = text.lower()
    # look for "0.2 m", "200 mm", "50 mm", "0.2M" (with or without space)
    m = re.search(r'(\d+\.?\d*)\s*(m|mm)\b', text)
    if not m:
        # Try without space requirement: "0.2M", "200mM"
        m = re.search(r'(\d+\.?\d*)(m|mm)\b', text)
        if not m:
            return None

    val, unit = m.groups()
    val = float(val)

    if unit == 'm':
        return val
    elif unit == 'mm':
        return val / 1000.0

    return None

def ionic_strength_from_condition(condition_text):
    """Compute ionic strength from condition text."""
    total_I = 0.0
    text_lower = condition_text.lower()

    # First, try to find all concentration-salt pairs in the text
    # Pattern: number + unit (M/mM) + optional space + salt name
    # This handles both "0.2 M Ammonium Sulfate" and "0.2M Ammonium Sulfate"
    # Pattern 1: with space after unit: "0.2 M potassium sodium tartrate"
    pattern_with_space = re.compile(r'(\d+\.?\d*)\s+(m|mm)\s+([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)', flags=re.I)
    # Pattern 2: without space after unit: "0.2M Ammonium Sulfate"
    pattern_no_space = re.compile(r'(\d+\.?\d*)(m|mm)([A-Za-z0-9\s\-/\(\)\.]+?)(?=[,;]|and|$)', flags=re.I)
    
    # Try pattern with space first
    for match in pattern_with_space.finditer(condition_text):
        val_str, unit, tail = match.groups()
        
        # Skip if tail contains 'PEG' (not a salt)
        if 'peg' in tail.lower():
            continue
        
        # Parse concentration
        v = float(val_str)
        if unit.lower() == 'mm':
            conc = v / 1000.0
        else:
            conc = v
        
        # Normalize salt name from tail (strip whitespace first)
        salt_name = normalize_name(tail.strip())
        if salt_name is None:
            continue  # ignore buffers, PEG, etc.

        z_plus, z_minus = SALTS[salt_name]

        # Ionic strength contribution:
        # I = 1/2 * C * (z+^2 + z-^2)
        I = 0.5 * conc * (z_plus**2 + z_minus**2)
        total_I += I
    
    # If no matches, try pattern without space requirement
    if total_I == 0.0:
        for match in pattern_no_space.finditer(condition_text):
            val_str, unit, tail = match.groups()
            
            # Skip if tail contains 'PEG' (not a salt)
            if 'peg' in tail.lower():
                continue
            
            # Parse concentration
            v = float(val_str)
            if unit.lower() == 'mm':
                conc = v / 1000.0
            else:
                conc = v
            
            # Normalize salt name from tail
            salt_name = normalize_name(tail)
            if salt_name is None:
                continue  # ignore buffers, PEG, etc.

            z_plus, z_minus = SALTS[salt_name]

            # Ionic strength contribution:
            # I = 1/2 * C * (z+^2 + z-^2)
            I = 0.5 * conc * (z_plus**2 + z_minus**2)
            total_I += I

    # Fallback: if no matches found, try splitting by comma/semicolon
    if total_I == 0.0:
        parts = re.split(r'[;,]', text_lower)
        for part in parts:
            conc = parse_concentration(part)
            if conc is None:
                continue

            salt_name = normalize_name(part)
            if salt_name is None:
                continue  # ignore buffers, PEG, etc.

            z_plus, z_minus = SALTS[salt_name]

            # Ionic strength contribution:
            # I = 1/2 * C * (z+^2 + z-^2)
            I = 0.5 * conc * (z_plus**2 + z_minus**2)
            total_I += I

    return total_I if total_I > 0.0 else None

def ionic_strength_category(ionic_strength):
    """Categorize ionic strength."""
    if ionic_strength is None or ionic_strength == 0:
        return 'Unknown'
    if ionic_strength >= 1.0:
        return '≥1.0 M'
    if ionic_strength >= 0.5:
        return '0.5-1.0 M'
    return '0-0.5 M'

def peg_category_from_percent(percent):
    if percent is None:
        return 'None'
    p = percent
    if p >= 30:
        return 'PEG (30-50%)'
    if p >= 20:
        return 'PEG (20-30%)'
    if p >= 10:
        return 'PEG (10-20%)'
    if p > 0:
        return 'PEG (0-10%)'
    return 'None'

def peg_mw_category(mw):
    if mw is None:
        return 'Unknown'
    mw = float(mw)
    if mw < 1000:
        return 'PEG 0-1k'
    if 1000 <= mw < 2000:
        return 'PEG 1k-2k'
    if 2000 <= mw < 10000:
        return 'PEG 2k-10k'
    if mw >= 10000:
        return 'PEG 10k-20k'
    return 'Unknown'

def parse_molarities(text):
    """
    Return list of (value_in_M, raw_match_text) found.
    Accepts formats '0.2 M', '200 mM' etc.
    """
    results = []
    for m in M_PATTERN.finditer(text):
        val_str, unit = m.groups()
        v = float(val_str)
        unit = unit.lower()
        if unit == 'mm' or unit == 'mm' or unit == 'mm':  # just to be safe
            # if pattern accidentally captured 'mM' we handle:
            # but regex captured 'mM' as 'mM' -> handle case-insensitively
            pass
        if m.group(2).lower() == 'mm':
            v_m = v / 1000.0
        else:
            v_m = v
        results.append((v_m, m.group(0)))
    return results

def find_peg_percent(text):
    """
    Find PEG percentage from various formats:
    - PEG 4K 20%
    - 22% (w/v) PEG 3350
    - 20% Polyethylene glycol 1000
    """
    # Look for PEG-specific percentage patterns
    peg_percent_patterns = [
        # PEG 4K 20% format
        re.compile(r'(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{1,2})\s*[Kk]\s*([0-9]{1,3}(?:\.[0-9]+)?)\s*%', flags=re.I),
        # 22% (w/v) PEG 3350 format
        re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:\(w/v\))?\s*(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)[-_\s]?([0-9]{2,5})', flags=re.I),
        # 20% Polyethylene glycol 1000 format
        re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*Polyethylene\s+glycol\s+([0-9]{2,5})', flags=re.I),
        # Generic PEG percentage (fallback)
        re.compile(r'([0-9]{1,3}(?:\.[0-9]+)?)\s*%\s*(?:.*?\s)?(?:PEG|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)', flags=re.I),
        re.compile(r'PEG.*?([0-9]{1,3}(?:\.[0-9]+)?)\s*%', flags=re.I)
    ]

    # PEG_MW_PATTERN = re.compile(
    #     r'(?:PEG|PEMME|polyethylene\s+glycol(?:\s+monomethyl\s+ether)?)'
    #     r'[-_\s]?([0-9]{1,3}(?:,[0-9]{3})*|[0-9]{2,5})(?=[^\d]|$)',
    #     flags=re.I
    # )

    for pattern in peg_percent_patterns:
        m = pattern.search(text)
        if m:
            try:
                # Extract the percentage value (usually the first group)
                percent_str = m.group(1)
                return float(percent_str)
            except (ValueError, IndexError):
                continue
    
    return None

def first_percent(text):
    m = PERCENT_PATTERN.search(text)
    if m:
        return float(m.group(1))
    return None

def find_peg_mw(text):
    # check PEG3350 style
    m = PEG_MW_PATTERN.search(text)
    if m:
        try:
            # return float(m.group(1))
            return float(m.group(1).replace(',', ''))
        except:
            pass
    # check 4K style
    m2 = PEG_K_PATTERN.search(text)
    if m2:
        try:
            return float(m2.group(1)) * 1000.0
        except:
            pass
    # check "Polyethylene glycol 1000" format
    m3 = re.search(r'Polyethylene\s+glycol\s+([0-9]{2,5})', text, flags=re.I)
    if m3:
        try:
            return float(m3.group(1))
        except:
            pass
    # sometimes 'PEG 4K' may be 'PEG 4K 20%' etc caught above
    return None


def get_exp_solution_data():
    """Get PDBId and exp_Solution data from database."""
    try:
        exp_table = config.get_reference_table_name()
        query = f"SELECT PDBId, exp_Solution FROM {exp_table} WHERE exp_Solution IS NOT NULL AND exp_Solution != ''"
        results = db.execute_query(query)
        
        entries = []
        for pdb_id, exp_solution in results:
            if exp_solution and exp_solution.strip():
                entries.append((pdb_id, exp_solution))
                
        logger.info(f"Retrieved {len(entries)} structures with exp_Solution data")
        return entries
        
    except Exception as e:
        logger.error(f"Error retrieving data from database: {str(e)}")
        return []

def classify_buffer_conditions(entries):
    """Classify buffer conditions for all entries."""
    classified_data = []
    
    for pdbid, detail in entries:
        try:
            text = detail.replace('\n',' ').strip()
            
            # Calculate ionic strength
            ionic_strength = ionic_strength_from_condition(text)
            ionic_strength_cat = ionic_strength_category(ionic_strength)
            
            # Parse PEG information
            peg_pct = find_peg_percent(text)
            peg_mw = find_peg_mw(text)
            # If either percent or MW is missing, treat as no PEG present
            if peg_pct is None and peg_mw is None:
                peg_cat = 'No PEG'
                peg_mw_cat = 'No PEG'
            else:
                peg_cat = peg_category_from_percent(peg_pct)
                peg_mw_cat = peg_mw_category(peg_mw)
            
            # Buffer detection: pick largest buffer molarity in text
            buffer_M = None
            buffer_matches = []
            buf_match = None

            # Case 1: molarity first ("0.1 M Tris-HCl")
            pattern_molarity_first = re.compile(
                r'([0-9]*\.?[0-9]+)\s*(mM|M)\s*([A-Za-z0-9\-/\s\(\)\.]*)',
                flags=re.I
            )

            # Case 2: buffer first ("Tris-HCl 0.1 M")
            pattern_buffer_first = re.compile(
                r'([A-Za-z0-9\-/\s\(\)\.]+?)\s*([0-9]*\.?[0-9]+)\s*(mM|M)',
                flags=re.I
            )

            # Helper to normalize concentration to molar (M)
            def normalize_conc(val_str, unit):
                v = float(val_str)
                return v / 1000.0 if unit.lower() == "mm" else v

            # Helper to check buffer keywords
            def find_buffer_keyword(text, buffers):
                text_low = text.lower()
                for buf in buffers:
                    if buf in text_low:
                        return buf
                return None

            buffer_matches = []

            # --- Match case 1: '0.1 M Tris-HCl' ---
            for match in pattern_molarity_first.finditer(text):
                val_str, unit, tail = match.groups()
                buf = find_buffer_keyword(tail, BUFFERS)
                if buf:
                    v = normalize_conc(val_str, unit)
                    buffer_matches.append((v, match.group(0)))
                    buf_match = buf
                    print(f"Case1 -> {val_str} {unit} {tail} | Matched buffer: {buf}")

            # --- Match case 2: 'Tris-HCl 0.1 M' ---
            for match in pattern_buffer_first.finditer(text):
                buf_name, val_str, unit = match.groups()
                buf = find_buffer_keyword(buf_name, BUFFERS)
                if buf:
                    v = normalize_conc(val_str, unit)
                    buffer_matches.append((v, match.group(0)))
                    buf_match = buf
                    print(f"Case2 -> {buf_name} {val_str} {unit} | Matched buffer: {buf}")
                        
            if buffer_matches:
                # choose highest buffer M found
                buffer_M = max(x[0] for x in buffer_matches)
                
            # buffer_cat = 'Unknown' if buffer_M is None else ('High' if buffer_M>=0.1 else 'Low')
            # print("buffer_M",buffer_M)
            buffer_cat = (
                'Unknown' if buffer_M is None else
                '0-0.1 M' if buffer_M < 0.1 else
                '0.1-0.5 M' if buffer_M < 0.5 else
                '0.5-5.0 M' if buffer_M < 5 else
                '>5.0 M'
            )
            classified_data.append({
                'PDBId': pdbid,
                'exp_Solution': text,
                'ionic_strength': ionic_strength if ionic_strength is not None else '',
                'ionic_strength_category': ionic_strength_cat if ionic_strength_cat else '',
                'PEG_concentration': peg_cat,
                'PEG_percent': peg_pct if peg_pct is not None else '',
                'PEG_MW_category': peg_mw_cat,
                'PEG_MW': int(peg_mw) if peg_mw else '',
                'buffer_category': buffer_cat,
                'buffer_M': buffer_M if buffer_M else '',
                'buffer_matches': buf_match if buf_match else '',
                'notes': ''
            })
            
        except Exception as e:
            logger.error(f"Error processing {pdbid}: {str(e)}")
            # Add entry with error information
            classified_data.append({
                'PDBId': pdbid,
                'exp_Solution': detail,
                'ion_category': 'Error',
                'est_total_salt_M': '',
                'ion_matches': '',
                'salt_type': '',
                'salt_concentration_category': '',
                'ion_valency': '',
                'PEG_concentration': 'Error',
                'PEG_percent': '',
                'PEG_MW_category': 'Error',
                'PEG_MW': '',
                'buffer_category': 'Error',
                'buffer_M': '',
                'notes': f'Processing error: {str(e)}'
            })
    
    return classified_data

def save_results(classified_data, output_file=None):
    """Save classified data to CSV file."""
    if output_file is None:
        # Save in parent folder from config
        parent_folder = config.get_parent_folder()
        output_file = Path(parent_folder) / 'classified_buffer_conditions.csv'
    else:
        output_file = Path(output_file)

    print("output_file", output_file)
    
    try:
        with output_file.open('w', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh)
            writer.writerow(['PDBId','exp_Solution','ionic_strength','ionic_strength_category',
                           'PEG_concentration','PEG_percent','PEG_MW_category','PEG_MW',
                           'buffer_category','buffer_M','buffer_matches','notes'])
            
            for data in classified_data:
                writer.writerow([
                    data['PDBId'], data['exp_Solution'], 
                    data['ionic_strength'], data['ionic_strength_category'],
                    data['PEG_concentration'], data['PEG_percent'], data['PEG_MW_category'], data['PEG_MW'],
                    data['buffer_category'], data['buffer_M'], data['buffer_matches'], data['notes']
                ])
        
        logger.info(f"Results saved to {output_file}")
        return str(output_file)
        
    except Exception as e:
        logger.error(f"Error saving results: {str(e)}")
        return None

def main():
    """Main function to classify buffer conditions from database."""
    logger.info("Starting buffer condition classification...")
    
    # Get data from database
    entries = get_exp_solution_data()
    
    if not entries:
        logger.warning("No exp_Solution data found in database")
        return
    
    # Classify buffer conditions
    logger.info("Classifying buffer conditions...")
    classified_data = classify_buffer_conditions(entries)
    
    # Save results
    output_file = save_results(classified_data)
    
    if output_file:
        logger.info(f"Successfully classified {len(classified_data)} structures")
        logger.info(f"Results saved to: {output_file}")
        
        # Show summary statistics
        ionic_strength_categories = [d['ionic_strength_category'] for d in classified_data if d['ionic_strength_category'] != 'Error' and d['ionic_strength_category']]
        peg_categories = [d['PEG_concentration'] for d in classified_data if d['PEG_concentration'] != 'Error']
        buffer_categories = [d['buffer_category'] for d in classified_data if d['buffer_category'] != 'Error']
        
        print(f"\nClassification Summary:")
        print(f"Total structures processed: {len(classified_data)}")
        print(f"Ionic strength categories: {dict(pd.Series(ionic_strength_categories).value_counts())}")
        print(f"PEG concentration categories: {dict(pd.Series(peg_categories).value_counts())}")
        print(f"Buffer concentration categories: {dict(pd.Series(buffer_categories).value_counts())}")
    else:
        logger.error("Failed to save results")

if __name__ == "__main__":
    main() 