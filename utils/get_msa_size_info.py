from pathlib import Path
import argparse
import pandas as pd
import json
from tqdm import tqdm

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def get_msa_info(json_path):
    data = read_json(json_path)
    name = data['name']
    samples = []
    for s in data['sequences']:
        for k, v in s.items():
            seq = s[k]['sequence']
            sample = {
                    'sequence': seq,
                    'has_msa': not s[k]['unpairedMsa'] == f'>query\n{seq}\n',
                    'chain_id': s[k]['id'],
                    'mol_type': k,
                    'pdb_id': name[:4],
                    'json_id': name,
                    'msa_length': int((len(str(s[k]['unpairedMsa']).split('\n')) - 1) / 2),
            }
            samples.append(sample)
    return pd.DataFrame(samples)

def main(json_dir):
    json_dir = Path(json_dir)
    # json_dir.mkdir(exist_ok=True, parents=True)

    data = []

    print(f'Start collecting json data from {json_dir.resolve()}')

    jsons = list(Path(json_dir).rglob('*_data.json'))

    print('done.')

    print('Found', len(jsons), 'jsons.')
    print('Start Processing')

    pbar = tqdm(total=len(jsons))

    for p in jsons:
        # print('Extract MSA info from:', p.stem)
        try:
            df = get_msa_info(p)
            data.append(df)
            pbar.update(1)
        except Exception as e:
            print(e)
            pbar.update(1)
            continue
    results = pd.concat(data)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Extract MSA size information from AlphaFold3 data JSON files'
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing *_data.json files'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV filename (default: msa_size_info.csv in the input folder)'
    )
    
    args = parser.parse_args()
    
    json_dir = Path(args.folder)
    
    if not json_dir.exists():
        print(f"Error: Folder does not exist: {json_dir}")
        exit(1)
    
    if not json_dir.is_dir():
        print(f"Error: Path is not a directory: {json_dir}")
        exit(1)
    
    results = main(json_dir)

    # print(results)
    print(results.groupby(['has_msa', 'mol_type']).count())

    # Output to the same folder
    if args.output:
        output_path = json_dir / args.output
    else:
        output_path = json_dir / "msa_size_info.csv"
    
    results.to_csv(output_path, index=False)
    print(f"\nOutput saved to: {output_path}")

