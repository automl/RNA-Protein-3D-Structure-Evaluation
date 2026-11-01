import filecmp
import os
import shutil
import sys
import subprocess
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # One level above utils package.
# Make the repo importable without exporting PYTHONPATH (macOS, Linux and Windows alike), and pass it
# on through the environment so the scripts started below (python ..., conda run -n ost ...) see it too.
if project_root not in sys.path:
    sys.path.insert(0, project_root)
_pythonpath = os.environ.get("PYTHONPATH", "")
if project_root not in _pythonpath.split(os.pathsep):
    os.environ["PYTHONPATH"] = project_root + (os.pathsep + _pythonpath if _pythonpath else "")

from database.startConfig import StartConfig

def execute_command(command, check=True):
    """Execute a shell command. A leading `python` runs in the same interpreter as this script
    (the active env), whatever `python` resolves to on PATH."""
    if command.startswith("python "):
        command = f'"{sys.executable}" ' + command[len("python "):]
    print(f"Executing: {command}")
    result = subprocess.run(command, shell=True, check=check)
    return result.returncode == 0

def process_folder(folder, pdb_command, af_command, extractInteractions_command, pdb_path, af_path, skip_interface=False):
    """Process PDB and AF folders."""
    if os.path.basename(folder) == "pdb":
        execute_command(f"python {pdb_command} {pdb_path}")
        if not skip_interface:
            execute_command(f"python {extractInteractions_command} {pdb_path}")

    if os.path.basename(folder) == "af":
        execute_command(f"python {af_command} {af_path}")
        if not skip_interface:
            execute_command(f"python {extractInteractions_command} {af_path}")

# --------------------------------------------------------------------------------------
# Working database: every run works on database/rbpDatabase.db. When a run ends it is COPIED to
# <folder>/rbpDatabase.db and stays in database/, so database/ always shows the database that was
# worked on last; its folder is the one recorded in database/config.json by that run.
# --------------------------------------------------------------------------------------
# EVAL_WORK_DB lets concurrent runs each use their own working database (pair it with
# EVAL_CONFIG_NAME, which startConfig.py already honours, so their config files are separate too).
WORK_DB = os.environ.get("EVAL_WORK_DB") or os.path.join(project_root, "database", "rbpDatabase.db")
DB_NAME = "rbpDatabase.db"
DSSR_HELP = """DSSR input for +dssr, one file per structure in {dssr}, named <id>_pdb_dssr.* / <id>_af_dssr.*,
made from the aligned RNA chains that -dssr writes to aligned/<id>_<pdb|af>_aligned_rna.pdb:
  best:      DSSR program -> JSON:  x3dna-dssr -i=aligned/<id>_pdb_aligned_rna.pdb --json -o=dssr/<id>_pdb_dssr.json
  otherwise: https://web.x3dna-dssr.org/analyze -> upload the aligned RNA file -> "3DNA output" -> save as
             dssr/<id>_pdb_dssr.txt (converted by evaluation/dssrTxt2json.py)
  no file:   the 3DNA download fails (HTTP 404) when DSSR finds no base pair; a structure without a JSON
             or text file is treated as single-stranded (an empty dssr/ folder: all single-stranded)."""
_OLD_MARKER = WORK_DB + ".folder"            # owner file of earlier versions; no longer written


def _save_to(folder):
    """Copy the working database to <folder>/rbpDatabase.db (previous copy kept as rbpDatabase.prev.db)."""
    dst = os.path.join(folder, DB_NAME)
    if os.path.exists(dst):
        if filecmp.cmp(WORK_DB, dst, shallow=False):
            return
        os.replace(dst, os.path.join(folder, "rbpDatabase.prev.db"))
    shutil.copy2(WORK_DB, dst)
    print(f"Working database copied to {dst}")


def checkout_db(folder, fresh, owner=None):
    """Make database/rbpDatabase.db the database of `folder`.

    Same folder as last time: keep working on database/rbpDatabase.db (it is the newest copy, e.g.
    after a script was run on it directly) -- except for -dssr, which builds a new database.
    Another folder: the working database is first saved to its own folder (nothing is lost), then
    replaced by <folder>/rbpDatabase.db or the set-level <folder>/../rbpDatabase.db (-dssr: new one).
    owner = the folder of the previous run (database/config.json before this run changed it).
    """
    if os.path.exists(_OLD_MARKER):
        os.remove(_OLD_MARKER)
    if owner is not None and not os.path.isdir(owner):
        owner = None
    if os.path.exists(WORK_DB):
        if owner is None:
            own_copy = os.path.join(folder, DB_NAME)
            if os.path.exists(own_copy) and filecmp.cmp(WORK_DB, own_copy, shallow=False):
                owner = folder                 # identical to this folder's database
            else:
                sys.exit(f"Error: it is not known which folder {WORK_DB} belongs to. Copy it to its folder "
                         f"(or delete it if its folder already holds it), then rerun.")
        if owner == folder and not fresh:
            print(f"Working database: continuing with {WORK_DB} ({folder}).")
            return
        if os.path.isdir(owner):
            _save_to(owner)                    # keep any work not yet copied to its folder
        os.remove(WORK_DB)
    if fresh:
        print(f"Working database: -dssr builds a new {WORK_DB}.")
    else:
        for src in (os.path.join(folder, DB_NAME), os.path.join(os.path.dirname(folder), DB_NAME)):
            if os.path.exists(src):
                shutil.copy2(src, WORK_DB)
                print(f"Working database: copied {src} -> {WORK_DB}")
                break
        else:
            sys.exit(f"Error: no {DB_NAME} in {folder} or its parent folder. Run -dssr first.")


def checkin_db(folder):
    """Copy the working database to <folder>; it also stays in database/ as the last one worked on."""
    if os.path.exists(WORK_DB):
        _save_to(folder)
        print(f"Last worked-on database: {WORK_DB} ({folder})")


def _n_models(folder):
    af = os.path.join(folder or "", "af")
    return len([f for f in os.listdir(af) if f.endswith("_model_0.cif")]) if os.path.isdir(af) else 0


def _all_single_chain(pdb_dir):
    """True when every reference mmCIF in pdb_dir has exactly one polymer chain (single-chain RNAs:
    no interface, so extractInteractions would fail and move the files to check_pdb/)."""
    from Bio.PDB.MMCIF2Dict import MMCIF2Dict
    cifs = [f for f in os.listdir(pdb_dir) if f.endswith(".cif") and not f.startswith("fold_")] if os.path.isdir(pdb_dir) else []
    if not cifs:
        return False
    for f in cifs:
        strands = MMCIF2Dict(os.path.join(pdb_dir, f)).get("_entity_poly.pdbx_strand_id", [])
        strands = [strands] if isinstance(strands, str) else strands
        if sum(len(s.split(",")) for s in strands) != 1:
            return False
    return True


def _confirm_folder(folder):
    """No --folder given: show the folder from database/config.json and ask whether to process it.
    Enter/y keeps it, a path replaces it, n aborts. Without a terminal the run stops (give --folder)."""
    print(f"\nNo --folder given. Folder in database/config.json:\n  {folder}  ({_n_models(folder)} AF models)")
    try:
        ans = input("Proceed with this folder? [Y = yes / n = abort / or type another folder]: ").strip()
    except EOFError:
        sys.exit("Error: no --folder given and no terminal to confirm the configured folder; rerun with --folder.")
    if ans.lower() in ("", "y", "yes"):
        return folder
    if ans.lower() in ("n", "no"):
        sys.exit("Aborted; rerun with --folder <processing folder>.")
    return ans


def add_msa_templates(folder, af_path):
    """MSA depth (addMSA_fromJSON.py) and template ids (extractTemplatesAF3.py) for the folder.

    Source: AlphaFold Server downloads anywhere below <folder>/folds (job_request + msas/ +
    templates/), converted to *_data.json in a temporary <folder>/folds/data_json that is removed
    afterwards (it only repeats what folds/ holds, ~4 GB per 100 jobs); otherwise local-AF3
    *_data.json files below af/.
    """
    folds = os.path.join(folder, "folds")
    tmp = None
    if os.path.isdir(folds):
        json_dir = tmp = os.path.join(folds, "data_json")
        execute_command(f"python {os.path.join(project_root, 'utils', 'serverFolds2dataJSON.py')} {folds} {json_dir}")
    elif any(f.endswith("_data.json") and "full_data" not in f for _, _, fs in os.walk(af_path) for f in fs):
        json_dir = af_path
    else:
        print(f"No AlphaFold Server folds/ or *_data.json found in {folder}; skipping MSA/template extraction.")
        return
    try:
        execute_command(f"python {os.path.join(project_root, 'utils', 'addMSA_fromJSON.py')} {json_dir}", check=False)
        execute_command(f"python {os.path.join(project_root, 'database', 'extractTemplatesAF3.py')} {json_dir}", check=False)
    finally:
        if tmp and os.path.isdir(tmp):
            shutil.rmtree(tmp)


def add_domains(folder):
    """InterPro protein domains of the exp table (database/fetchInterProDomains.py: UniProt id of each protein
    chain -> InterPro domains/repeats, kept when present in the deposited construct). Answers are cached in
    <folder>/domains/interpro_cache.json, so reruns do not query InterPro again."""
    execute_command(f"python {os.path.join(project_root, 'database', 'fetchInterProDomains.py')} "
                    f"{os.path.join(folder, 'domains')}", check=False)


def main():
    parser = argparse.ArgumentParser(
        description='Main execution script for evaluating protein-RNA and RNA structures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
DSSR (+dssr): JSON from the DSSR program is used as it is; otherwise save the DSSR web server's
"3DNA output" (https://web.x3dna-dssr.org/analyze, input aligned/<id>_<pdb|af>_aligned_rna.pdb) as
dssr/<id>_<pdb|af>_dssr.txt; structures with neither are single-stranded (no base pair -> the server
returns 404 and there is no file).

The processing folder (--folder; without it, the folder in database/config.json, which the run asks
you to confirm or replace before it starts) holds af/, pdb/,
optional folds/ (AlphaFold Server downloads: MSAs and templates), dssr/ and the outputs (aligned/,
lddt/, ...). Its last path component sets the tables (protein_rna -> pred_/exp_protein_rna,
rna_rna -> pred_/exp_rna_rna). Plots are written to <folder>/plots and the table exports to
<folder>/<table>.csv. The run works on
database/rbpDatabase.db and copies it to
<folder>/rbpDatabase.db when it ends; it stays in database/ (database/config.json names its
folder). A run on another folder first saves it to its own folder, then copies the new folder's
database in (-dssr builds a new one).

Examples:
  # New folder: parse, score, MSA/templates (from folds/), InterPro domains, OST, SWISS-MODEL interfaces
  python mainExecution.py -dssr --folder data/2026/2026_a2021nonX/protein_rna
  # Only add MSA depth and templates to an existing folder database
  python mainExecution.py --addMSAtemp --folder data/2026/2026_a2021nonX/protein_rna
  # Only add InterPro protein domains to an existing folder database (or both: --addMSAtemp --addDomains)
  python mainExecution.py --addDomains --folder data/2026/2026_a2021nonX/protein_rna
  # Process without DSSR, skip interface extraction
  python mainExecution.py -dssr -interface
  # Process without DSSR, skip AF3 metrics
  python mainExecution.py -dssr -af3
  # Process with DSSR, include MSA processing, with database updates
  python mainExecution.py +dssr -addMSA --db-update
  # Process with DSSR, include MSA processing, without database updates
  python mainExecution.py +dssr -addMSA
  # Process with DSSR, include plotting with MSA and RBP template filters
  python mainExecution.py +dssr -addMSA --plot all +msa --rbp +rbptemp
  # Process with DSSR, generate plots with ALL combinations of filters
  python mainExecution.py +dssr -addMSA --plot all +msa --allArgs
  # Process with DSSR, generate plots with ALL combinations (folder from database/config.json, confirmed at the prompt)
  python mainExecution.py +dssr --allArgs
        """
    )

    parser.add_argument('--folder', default=None,
                        help='Processing folder with af/, pdb/ (and folds/, dssr/), e.g. '
                             'data/2026/2026_a2021nonX/protein_rna. Default: the folder in database/config.json, '
                             'which the run asks you to confirm (or replace) before it starts.')
    parser.add_argument('--addMSAtemp', dest='add_msa_temp', action='store_true',
                        help='Only add MSA depth and template ids (from <folder>/folds or *_data.json) '
                             'to the folder database, then stop.')
    parser.add_argument('--addDomains', dest='add_domains', action='store_true',
                        help='Only add InterPro protein domains (domain_ids/_names/_counts/_pos of the exp '
                             'table, via each chain\'s UniProt id; database/fetchInterProDomains.py), then stop. '
                             'Can be combined with --addMSAtemp.')
    parser.add_argument('dssr_flag', nargs='?', choices=['+dssr', '-dssr'],
                       help='+dssr: Process DSSR evaluation, -dssr: Skip DSSR evaluation (can also be specified as +dssr or -dssr anywhere in arguments). Default: -dssr')

    # Options for -dssr
    parser.add_argument('-pdbparse', dest='skip_pdb_parse', action='store_true',
                       help='Skip the pdb/ branch: no parsePDBCIF2DB, extractInteractions, '
                            'extractInterfaceArea or estimateBindingAffinity on the reference '
                            'structures, and the folder database is kept instead of rebuilt. Use '
                            'when <folder>/rbpDatabase.db already holds the exp_ table (only for -dssr).')
    parser.add_argument('-interface', action='store_true',
                       help='Skip extractInteractions, extractInterfaceArea, estimateBindingAffinity (only for -dssr)')
    parser.add_argument('-af3', action='store_true',
                       help='Skip af3Metrics_new (only for -dssr)')
    basic_group = parser.add_argument_group(
        'basicEval options',
        'Minimal -dssr pipeline. On pred_protein_rna, ost_RBPinterfaces scores the interfaces with '
        'the SWISS-MODEL service when a token is given (--token or SWISSMODEL_OST_TOKEN), otherwise '
        'with the local OpenStructure installation; -noRBP skips it.')
    basic_group.add_argument('-basicEval', dest='basic_eval', action='store_true',
                       help='Minimal pipeline (-dssr only): parsePDBCIF2DB, parseAFcif2DB, '
                            'extractEvalMetrics and the OST scoring scripts (all-atom LDDT, RNA '
                            'AllChains, RBP interfaces). Skips extractInteractions, interface area, '
                            'binding affinity and af3Metrics.')
    basic_group.add_argument('--token', dest='token', metavar='TOKEN', default=None,
                       help='SWISS-MODEL API token for ost_RBPinterfaces (overrides '
                            'SWISSMODEL_OST_TOKEN). Create/regenerate at '
                            'https://swissmodel.expasy.org/account')
    basic_group.add_argument('-noAllChains', dest='skip_allchains', action='store_true',
                       help='Skip ost_calcRNA_AllChains (RNA_LDDT/iLDDT/ICS_AllChains). '
                            'Useful when every complex has a single RNA chain, where the per-chain '
                            'LDDT adds nothing over RNA_LDDT_allAtom (only for -dssr).')
    parser.add_argument('-noDomains', dest='skip_domains', action='store_true',
                       help='Skip the InterPro domain annotation of a -dssr run on pred_protein_rna '
                            '(database/fetchInterProDomains.py; needs internet). Also skipped by -basicEval.')
    basic_group.add_argument('-noRBP', dest='skip_rbp', action='store_true',
                       help='Skip ost_RBPinterfaces and suppress the SWISS-MODEL token prompt. '
                            'Only relevant for -dssr on pred_protein_rna.')

    # Options for +dssr
    parser.add_argument('-addMSA', dest='msa', action='store_true',
                        help='+dssr: add MSA depth and templates (same step as --addMSAtemp)')
    parser.add_argument('--db-update', dest='db_update', action='store_true',
                       help='Run addPH, extractElectrostaticPotential, phCorrelation.py db_update (only for +dssr)')

    # Plotting options
    parser.add_argument('--plot', nargs='*', metavar=('MSA_OPTION', 'CHAIN_OPTION'),
                       help='Optional: Generate plots. Format: --plot <+msa|-msa> <all|single_chain>. If not specified with +dssr, defaults to "all".')
    parser.add_argument('--rbp', choices=['+rbptemp', '-rbptemp', '+rbp', '-rbp'],
                       help='Optional: RBP template filter. Use with --plot. Format: +rbptemp or -rbptemp. '
                            'Note: Use --rbp=-rbptemp or --rbp=+rbptemp to avoid argparse issues with negative values.')
    parser.add_argument('--allArgs', action='store_true',
                       help='Optional: Generate plots with all combinations of filters (works with or without --plot). '
                            'Excludes "all" combination. Combinations: all/single_chain × +msa/-msa × +rbptemp/-rbptemp')

    # Pre-process arguments to handle --rbp with negative values and -dssr/+dssr
    # argparse has issues with --rbp -rbptemp because -rbptemp looks like a flag
    # Also, -dssr looks like a flag but is actually a positional argument
    # Convert --rbp -rbptemp to --rbp=-rbptemp
    # Convert -dssr to dssr_flag=-dssr (as a workaround, we'll handle it in code)
    processed_argv = []
    i = 0
    while i < len(sys.argv):
        if sys.argv[i] == '--rbp' and i + 1 < len(sys.argv):
            next_arg = sys.argv[i + 1]
            # If next argument starts with - and is a valid rbp choice, combine them
            if next_arg in ['+rbptemp', '-rbptemp', '+rbp', '-rbp']:
                processed_argv.append(f'--rbp={next_arg}')
                i += 2
                continue
        # Handle -dssr and +dssr - they can appear anywhere
        if sys.argv[i] in ['-dssr', '+dssr']:
            # Skip it here, we'll handle it separately
            i += 1
            continue
        processed_argv.append(sys.argv[i])
        i += 1
    
    # Replace sys.argv temporarily for parsing
    original_argv = sys.argv
    sys.argv = processed_argv
    
    # Use parse_known_args to avoid errors with -dssr
    args, unknown = parser.parse_known_args()
    
    # Restore original sys.argv
    sys.argv = original_argv
    
    # Parse +dssr/-dssr from arguments (can be anywhere)
    dssr_flag_found = None
    
    for arg in sys.argv[1:]:
        if arg in ['+dssr', '-dssr']:
            dssr_flag_found = arg

    # Set dssr_flag (defaults to -dssr when neither +dssr nor -dssr is given)
    if args.dssr_flag:
        dssr_flag = args.dssr_flag
    elif dssr_flag_found:
        dssr_flag = dssr_flag_found
    else:
        dssr_flag = '-dssr'
        if not (args.add_msa_temp or args.add_domains):
            print("No +dssr/-dssr given; defaulting to -dssr.")

    # Processing folder: --folder (written to database/config.json, which every script reads) or the
    # folder already configured there, which must be confirmed (or replaced) at the prompt, so a run
    # never silently processes the previous run's folder. The working database is always
    # database/rbpDatabase.db.
    config = StartConfig()
    previous_folder = config.get_parent_folder()      # owner of database/rbpDatabase.db (last run)
    if not args.folder:
        args.folder = _confirm_folder(previous_folder)
    if args.folder:
        folder = os.path.abspath(os.path.expanduser(args.folder))
        if not os.path.isdir(folder):
            sys.exit(f"Error: --folder {folder} does not exist.")
        if os.path.basename(folder) not in ("protein_rna", "rna_rna"):
            print(f"Warning: folder name '{os.path.basename(folder)}' sets the tables "
                  f"pred_{os.path.basename(folder)} / exp_{os.path.basename(folder)}.")
        config.set_config(parent_folder=folder)
    config.db_path = WORK_DB
    config.save_config()
    parent_folder = config.get_parent_folder()
    # Plots of this run go to <folder>/plots (plots/plotCreator.py and the other plotting scripts read
    # AF3EVAL_PLOTS_DIR via utils/plots_dir.py); scripts run on their own still use results/plots.
    os.environ["AF3EVAL_PLOTS_DIR"] = os.path.join(parent_folder, "plots")
    if not parent_folder:
        print("Error: parent_folder not found in startConfig.")
        print("Please ensure startConfig is properly configured.")
        sys.exit(1)
    print(f"Processing folder: {parent_folder}; {_n_models(parent_folder)} AF models")

    # Update args.dssr_flag for use in the rest of the code
    args.dssr_flag = dssr_flag

    # Validate argument combinations
    if args.dssr_flag == '-dssr' and (args.msa or args.db_update):
        print("Warning: +msa and +db_update options are only valid with +dssr. Ignoring them.")
        args.msa = False
        args.db_update = False

    if args.dssr_flag == '+dssr' and (args.interface or args.af3 or args.basic_eval):
        print("Warning: -interface, -af3 and -basicEval options are only valid with -dssr. Ignoring them.")
        args.interface = False
        args.af3 = False
        args.basic_eval = False

    # -basicEval is a minimal run: reuse the existing skip switches so the interface- and
    # af3-related steps are dropped, leaving parse -> extractEvalMetrics -> OST scoring.
    if args.basic_eval:
        print("basicEval: running minimal pipeline (parse -> extractEvalMetrics -> OST scoring).")
        args.interface = True
        args.af3 = True
        args.skip_domains = True

    # Single-chain RNAs have no interface: skip the interface steps as with -interface
    if (args.dssr_flag == '-dssr' and not args.interface and os.path.basename(parent_folder) == "rna_rna"
            and _all_single_chain(os.path.join(parent_folder, "pdb"))):
        print("All reference structures in pdb/ are single-chain RNAs: skipping extractInteractions, "
              "extractInterfaceArea and estimateBindingAffinity (as with -interface).")
        args.interface = True

    # Validate --plot arguments
    if args.plot is not None:
        if len(args.plot) == 0:
            # --plot was specified but no arguments provided
            args.plot = None
        elif len(args.plot) == 1:
            # Only chain option provided (MSA is optional)
            # args.plot stays as [chain_opt]
            pass
        elif len(args.plot) == 2:
            # Both MSA and chain options provided
            # args.plot stays as [msa_opt, chain_opt]
            pass
        elif len(args.plot) > 2:
            print("Warning: --plot accepts maximum 2 arguments. Using first 2.")
            args.plot = args.plot[:2]

    # Validate --rbp requires --plot
    if args.rbp and not args.plot:
        print("Warning: --rbp option is only valid with --plot. Ignoring --rbp.")
        args.rbp = None

    database_path = "database"
    eval_path = "evaluation"
    utils_path = "utils"

    pdb_path = config.ref_folder
    af_path = config.pred_folder
    _subs = ["af"] if args.skip_pdb_parse else ["pdb", "af"]
    subfolders = [os.path.join(parent_folder, subfolder) for subfolder in _subs]
    dssr_path = os.path.join(parent_folder, 'dssr')

    # Acquire the SWISS-MODEL token up front (before running anything) so a long -dssr run
    # isn't left waiting for input at the very end. Only relevant when ost_RBPinterfaces will
    # run: a -dssr pass on pred_protein_rna. An empty response scores locally later.
    swissmodel_token = None
    if (args.dssr_flag == '-dssr' and not (args.add_msa_temp or args.add_domains) and config.pred_table == "pred_protein_rna"
            and not args.skip_rbp):
        # Priority: --token > SWISSMODEL_OST_TOKEN env var > interactive prompt.
        swissmodel_token = (args.token or os.environ.get("SWISSMODEL_OST_TOKEN", "")).strip()
        if not swissmodel_token:
            try:
                swissmodel_token = input(
                    "\nost_RBPinterfaces needs a SWISS-MODEL API token (passed as --token).\n"
                    "Token: set SWISSMODEL_OST_TOKEN (or pass --token). Create/regenerate at your\n"
                    "SWISS-MODEL account: https://swissmodel.expasy.org/account\n"
                    "Enter token (or press Enter to score the interfaces with the local OpenStructure): "
                ).strip()
            except EOFError:
                swissmodel_token = ""

    fresh = (args.dssr_flag == '-dssr' and not (args.add_msa_temp or args.add_domains)
             and not args.skip_pdb_parse)
    checkout_db(parent_folder, fresh, owner=previous_folder)
    try:
        run(args, config, parent_folder, pdb_path, af_path, subfolders, dssr_path, swissmodel_token,
            database_path, eval_path, utils_path)
    finally:
        checkin_db(parent_folder)


def run(args, config, parent_folder, pdb_path, af_path, subfolders, dssr_path, swissmodel_token,
        database_path, eval_path, utils_path):
    if args.add_msa_temp or args.add_domains:
        if args.add_msa_temp:
            print(f"Adding MSA depth and templates for {parent_folder} ...")
            add_msa_templates(parent_folder, af_path)
        if args.add_domains:
            print(f"Adding InterPro domains for {parent_folder} ...")
            add_domains(parent_folder)
        return

    try:
        if args.dssr_flag == '+dssr':
            if not os.path.exists(dssr_path):
                print(f"Warning: DSSR folder not found at {dssr_path}")
                print(DSSR_HELP.format(dssr=dssr_path))
                return

            print(f"Processing with DSSR: {dssr_path}")
            # Real DSSR JSON is used as it is; for structures without one, a DSSR-style JSON is built
            # from the web server's 3DNA text, or as single-stranded when there is no text either.
            execute_command(f"python {os.path.join(project_root, eval_path, 'dssrTxt2json.py')} {parent_folder}")

            # MSA depth + templates (from folds/ or *_data.json); also part of --db-update
            if args.msa or args.db_update:
                add_msa_templates(parent_folder, af_path)


            # Database updates section
            if args.db_update:
                # With db_update: run addPH, extractElectrostaticPotential, phCorrelation.py db_update
                execute_command(
                    f"python {os.path.join(project_root, eval_path, 'dssrConformation.py')} {dssr_path}")
                execute_command(
                            f"python {os.path.join(project_root, eval_path, 'extractRnaMetrics.py')} {dssr_path}")
                execute_command(
                            f"python {os.path.join(project_root, eval_path, 'extractDSSRpairs.py')} {dssr_path}")
                execute_command(
                            f"python {os.path.join(project_root, eval_path, 'extractSecondaryFeat.py')} {dssr_path}")
                execute_command(
                    f"python {os.path.join(project_root, database_path, 'addPH.py')}")
                execute_command(
                    f"python {os.path.join(project_root, eval_path, 'extractElectrostaticPotential.py')}")
                execute_command(
                    f"python {os.path.join(project_root, eval_path, 'extractGCcomplexity.py')}")
                execute_command(
                    f"python {os.path.join(project_root, eval_path, 'phCorrelation.py')} db_update")
                execute_command(
                    f"python {os.path.join(project_root, database_path, 'add_ions_ligands_to_db.py')}")
                execute_command(
                            f"python {os.path.join(project_root, database_path, 'addExpDetails.py')}")

            # Common plots (only run if --plot is NOT specified)
            if not args.plot:
                execute_command(
                            f"python {os.path.join(project_root, eval_path, 'electrostaticPotentialPlots.py')} all")
                execute_command(
                        f"python {os.path.join(project_root, eval_path, 'correlationAnalysis.py')} --msa --length --complexity --gc --experiment --rna_family") #--date
                execute_command(
                    f"python {os.path.join(project_root, eval_path, 'spiderPlotGenerator.py')}")
                if config.pred_table == "pred_protein_rna":
                    execute_command(
                                    f"python {os.path.join(project_root, eval_path, 'plot_template_metrics.py')}")
                    execute_command(
                                f"python {os.path.join(project_root, eval_path, 'analyzeRNAFamilyMSA_singleDB.py')}")

            # Optional plotting with MSA/chain/RBP filters
            # Helper function to run plotting scripts
            def run_plotting_scripts(chain_opt, msa_opt=None, rbp_opt=None):
                """Run all plotting scripts with given filter options.
                Continues execution even if individual scripts fail (e.g., insufficient samples).
                """
                plot_args_parts = [chain_opt]
                if msa_opt:
                    plot_args_parts.append(msa_opt)
                if rbp_opt:
                    plot_args_parts.append(rbp_opt)
                plot_args = " ".join(plot_args_parts)

                # List of plotting scripts to run
                plotting_scripts = []
                if not args.interface:
                    plotting_scripts.extend([
                        ('interfaceMetrics.py', plot_args),
                        ('motifMetrics.py', plot_args)
                    ])
                plotting_scripts.extend([
                    ('secondaryFeatPlots.py', plot_args),
                    ('evaluationMetrics.py', plot_args),
                    ('plotResults.py', plot_args),
                    ('phCorrelation.py', plot_args),
                    ('plotIonsLigands.py', plot_args),
                    ('peg_analysis_plots.py', plot_args)
                ])
                print("plotting_scripts", plotting_scripts)

                # Run each script, continuing even if one fails
                for script_name, script_args in plotting_scripts:
                    script_path = os.path.join(project_root, eval_path, script_name)
                    try:
                        success = execute_command(
                            f"python {script_path} {script_args}",
                            check=False  # Don't raise exception on failure
                        )
                        if not success:
                            print(f"Warning: {script_name} failed (may be due to insufficient samples). Continuing...")
                    except Exception as e:
                        print(f"Warning: {script_name} raised an exception: {e}. Continuing...")
                        continue

            if args.allArgs:
                # Generate all combinations INCLUDING "all" and "single_chain" (without filters)
                # "all" by itself (no filters) = 1
                # "single_chain" by itself (no filters) = 1
                # 2 (chain) × 2 (msa) without RBP = 4
                # 2 (chain) × 2 (msa) × 2 (rbp) with RBP = 8
                # Total: 14 combinations (including "all" and "single_chain" by themselves)
                chain_options = ['all', 'single_chain']
                msa_options = ['-msa', '+msa']
                rbp_options = ['-rbptemp', '+rbptemp']

                combinations = []
                # First: "all" and "single_chain" by themselves (no filters)
                combinations.append(('all', None, None))
                combinations.append(('single_chain', None, None))
                # Then: combinations without RBP filter (4)
                for chain in chain_options:
                    for msa in msa_options:
                        combinations.append((chain, msa, None))
                # Finally: combinations with RBP filter (8)
                for chain in chain_options:
                    for msa in msa_options:
                        for rbp in rbp_options:
                            combinations.append((chain, msa, rbp))

                print(f"\nGenerating plots with all {len(combinations)} combinations (including 'all' and 'single_chain' by themselves)...")
                print("="*60)

                for chain_opt, msa_opt, rbp_opt in combinations:
                    # Build display name
                    display_name_parts = [chain_opt]
                    if msa_opt:
                        display_name_parts.append(msa_opt)
                    if rbp_opt:
                        display_name_parts.append(rbp_opt)
                    display_name = " / ".join(display_name_parts) if len(display_name_parts) > 1 else chain_opt
                    print(f"\nProcessing: {display_name}")
                    print("-"*60)

                    run_plotting_scripts(chain_opt, msa_opt, rbp_opt)

                print("\n" + "="*60)
                print(f"Completed all {len(combinations)} combinations!")
                print("="*60)
            elif args.plot:
                # Single combination from --plot and optional --rbp
                # Handle both cases: [chain_opt] or [msa_opt, chain_opt]
                if len(args.plot) == 1:
                    # Only chain option provided
                    chain_opt = args.plot[0]
                    msa_opt = None
                else:
                    # Both MSA and chain options provided
                    msa_opt, chain_opt = args.plot
                
                rbp_opt = args.rbp if args.rbp else None

                # Validate chain option
                if chain_opt not in ['all', 'single_chain']:
                    print(f"Warning: Invalid chain option '{chain_opt}'. Must be 'all' or 'single_chain'. Skipping plots.")
                # Validate MSA option if provided
                elif msa_opt is not None and msa_opt not in ['+msa', '-msa', '']:
                    print(f"Warning: Invalid MSA option '{msa_opt}'. Must be '+msa', '-msa', or empty. Skipping plots.")
                else:
                    # Build display name
                    display_name_parts = [chain_opt]
                    if msa_opt:
                        display_name_parts.append(msa_opt)
                    if rbp_opt:
                        display_name_parts.append(rbp_opt)
                    display_name = " / ".join(display_name_parts)
                    
                    print(f"\nGenerating plots with: {display_name}")
                    print("="*60)
                    
                    run_plotting_scripts(chain_opt, msa_opt if msa_opt else None, rbp_opt)
            else:
                # Default: run plots with "all" (no filters)
                print("\nGenerating plots with default: all (no filters)")
                print("="*60)
                run_plotting_scripts('all')
            
        else:
            # -dssr processing
            print("Processing without DSSR...")

            # Process folders
            for folder in subfolders:
                print(f"Processing folder: {folder}")
                pdb_command = None
                af_command = None
                extractInteractions_command = None

                if os.path.basename(folder) == "pdb":
                    pdb_command = os.path.join(project_root, database_path, "parsePDBCIF2DB.py")
                    extractInteractions_command = os.path.join(project_root, eval_path, "extractInteractions.py")
                elif os.path.basename(folder) == "af":
                    af_command = os.path.join(project_root, database_path, "parseAFcif2DB.py")
                    extractInteractions_command = os.path.join(project_root, eval_path, "extractInteractions.py")

                process_folder(folder, pdb_command, af_command, extractInteractions_command,
                              pdb_path, af_path, skip_interface=args.interface)

            # MSA depth + templates, as soon as the predictions are in the database
            add_msa_templates(parent_folder, af_path)
            # InterPro protein domains of the reference structures (protein-RNA only)
            if config.pred_table == "pred_protein_rna" and not args.skip_domains:
                add_domains(parent_folder)

            # Process interface-related metrics (unless -interface flag)
            for folder in subfolders:
                if os.path.basename(folder) == "pdb" and not args.interface:
                    # continue
                    # execute_command(f"python {os.path.join(project_root, database_path, "insert_proteinFamily2DB.py")
                    execute_command(f"python {os.path.join(project_root, eval_path, 'extractInterfaceArea.py')} {pdb_path}")
                    execute_command(
                        f"python {os.path.join(project_root, eval_path, 'estimateBindingAffinity.py')} {config.ref_table}")

                if os.path.basename(folder) == "af":
                    if not args.interface:
                        execute_command(
                            f"python {os.path.join(project_root, eval_path, 'extractInteractions.py')} {af_path}")
                        execute_command(f"python {os.path.join(project_root, eval_path, 'extractInterfaceArea.py')} {af_path}")
                        execute_command(
                            f"python {os.path.join(project_root, eval_path, 'estimateBindingAffinity.py')} {config.pred_table}")

                    if not args.af3:
                        # delete --no-plots if plots for each sample required
                        execute_command(f"python {os.path.join(project_root, eval_path, 'af3Metrics_new.py')} {af_path} --no-plots")

                    # extractEvalMetrics takes no positional args (table/paths come from config.json).
                    # It also computes the backbone (C3'/CA) LDDT columns Protein_LDDT / RNA_LDDT /
                    # Complex_LDDT of the published databases; the all-atom LDDT comes from the OST
                    # scripts below. -lddt would skip the backbone LDDT.
                    execute_command(
                        f"python {os.path.join(project_root, eval_path, 'extractEvalMetrics.py')}")
                    # Models with many ions carry multi-letter chain ids (AA, AB, ...) that cannot be
                    # written in PDB format, so US-align and the all-atom LDDT get no aligned file for
                    # them. Rename the chains first: python dataPrep/rename_ion_chains.py (see its header).

                    # OST-based scoring runs in the 'ost' conda env. `conda run -n ost` executes
                    # each script in that env and returns to the ambient RNABench env afterwards,
                    # so the envs stay separate (no need to combine them). PYTHONPATH must point at
                    # the repo root so the ost_* scripts resolve their package-style imports.
                    print("\n" + "="*60)
                    print("Running OST-based scoring (All-Atom LDDT / RNA AllChains / RBP interfaces)...")
                    print("Note: these run in the 'ost' conda env via `conda run -n ost`.")
                    print("="*60)

                    is_protein_rna = config.pred_table == "pred_protein_rna"

                    def run_ost(script_name, script_args=""):
                        script_path = os.path.join(project_root, eval_path, script_name)
                        if not os.path.exists(script_path):
                            print(f"Warning: {script_path} not found. Skipping.")
                            return
                        print(f"\nRunning {script_name} in 'ost' env...")
                        # --no-capture-output: stream the child's stdout/stderr live. Without it,
                        # `conda run` buffers all output (incl. tqdm bars) and dumps it only on exit.
                        # PYTHONPATH=<repo> is inherited from os.environ (set at the top of this file).
                        command = f"conda run --no-capture-output -n ost python {script_path} {script_args}".strip()
                        execute_command(command, check=False)

                    # All-atom LDDT: valid for both pred_protein_rna and pred_rna_rna.
                    run_ost("ost_extractAllAtomLDDT.py", config.pred_table)

                    # RNA AllChains: interface metrics only make sense for protein-RNA. For
                    # pred_rna_rna run LDDT only (--lddt); for pred_protein_rna run the full
                    # LDDT + interface pass (no flag).
                    if args.skip_allchains:
                        print("Skipping ost_calcRNA_AllChains (-noAllChains).")
                    elif is_protein_rna:
                        run_ost("ost_calcRNA_AllChains.py")
                    else:
                        run_ost("ost_calcRNA_AllChains.py", "--lddt")

                    # RBP interfaces: protein-RNA only. The SWISS-MODEL token was acquired up
                    # front (swissmodel_token). With a token the interfaces are scored by the
                    # SWISS-MODEL OpenStructure service (as for the paper); without one, by the same
                    # OpenStructure action run locally in the 'ost' env (--local). -noRBP skips it.
                    if args.skip_rbp:
                        print("Skipping ost_RBPinterfaces (-noRBP).")
                    elif is_protein_rna:
                        if swissmodel_token:
                            # Pass the token through the environment, not the command line, so it is
                            # not echoed in logs or in conda's error message.
                            os.environ["SWISSMODEL_OST_TOKEN"] = swissmodel_token
                            run_ost("ost_RBPinterfaces.py")
                        else:
                            print("No SWISS-MODEL token provided. Scoring the interfaces with the local "
                                  "OpenStructure installation.")
                            run_ost("ost_RBPinterfaces.py", "--local")
                    else:
                        print(f"Skipping ost_RBPinterfaces (only valid for pred_protein_rna; "
                              f"current table: {config.pred_table}).")



    except FileNotFoundError as e:
        print(f"Error: {e}")
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

    # Export to CSV
    print("\n" + "="*60)
    print("Exporting database to CSV...")
    print("="*60)
    csv_command = os.path.join(project_root, database_path, "exportDB2CSV.py")
    execute_command(f'python {csv_command} {config.pred_table} "{parent_folder}"', check=False)
    execute_command(f'python {csv_command} {config.ref_table} "{parent_folder}"', check=False)

    # Final message
    print("\n" + "="*60)
    if args.dssr_flag == '-dssr':
        print(f"Processing complete!")
        print(f"\nTo continue with DSSR evaluation, add DSSR files to:")
        print(f"  {dssr_path}")
        print(f"\nThen run:")
        print(f"  python {os.path.basename(__file__)} +dssr --folder {parent_folder}")
    else:
        print("Processing complete!")
    print("="*60)

if __name__ == "__main__":
    main()
