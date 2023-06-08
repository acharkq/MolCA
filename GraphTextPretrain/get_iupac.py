import pubchempy as pcp
from pathlib import Path
from tqdm import tqdm

subsets = ['train', 'valid', 'test']
for sub in subsets:
    path = Path('data/PubChemDataset_v4/%s/smiles' % sub)
    files = list(path.glob('*'))
    files.sort()
    target_dir = Path('iupac/%s/' % sub)
    target_dir.mkdir(exist_ok=True)
    target_smiles = target_dir / 'smiles.txt'
    if not target_smiles.exists():
        smiles_list = []
        for file in tqdm(files):
            with open(file, 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines]
                lines = [line for line in lines if line]
            assert len(lines) == 1
            smiles = lines[0]
            smiles_list.append(smiles)
        with open(target_smiles, 'w') as f:
            for smiles in smiles_list:
                f.write(f'{smiles}\n')
    else:
        with open(target_smiles, 'r') as f:
            lines = f.readlines()
            smiles_list = [line.strip() for line in lines]

    target_path = Path('iupac/%s/iupac.txt' % sub)
    if target_path.exists():
        with open(target_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines if line.strip()]
    passed = len(lines)
    smiles_list = smiles_list[passed:]
    # print(files)
    with open(target_path, 'a', encoding='utf-8') as ff:
        for smiles in tqdm(smiles_list):
            try:
                compounds = pcp.get_compounds(smiles, namespace='smiles') 
            except pcp.BadRequestError as e:
                print(e)
                continue
            match = compounds[0]
            cid = compounds
            iupac_name = match.iupac_name
            line = f'{smiles}\t{cid}\t{iupac_name}\n'
            ff.write(line)
            # break