from pathlib import Path
import json

import torch
from tqdm import tqdm
from ogb.utils import smiles2graph
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.loader.dataloader import Collater
from itertools import repeat


def smiles2data(smiles):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

class ReactionCollater:
    def __init__(self, tokenizer, text_max_len, ph_token_id):
        self.collater = Collater([], [])
        self.tokenizer = tokenizer
        self.cls_id = self.tokenizer.cls_token_id
        self.text_max_len = text_max_len
        self.ph_token_id = ph_token_id
    
    def __call__(self, data_list):
        reaction_list, notes_list, mols_list = zip(*data_list)
        
        ## deal with text_list
        sentence_tokens = self.tokenizer(text=reaction_list,
                                         truncation=False,
                                         padding='longest',
                                         add_special_tokens=True,
                                         # max_length=self.text_max_len,
                                         return_tensors='pt',
                                         return_attention_mask=True, 
                                         return_token_type_ids=False)
        is_ph_token = (sentence_tokens.input_ids == self.ph_token_id) # shape = [N, 2]
        sentence_tokens['is_ph_token'] = is_ph_token
        
        ## deal with notes list
        notes_tokens = self.tokenizer(text=notes_list,
                                      truncation=True,
                                      padding='max_length',
                                      add_special_tokens=True,
                                      max_length=self.text_max_len,
                                      return_tensors='pt',
                                      return_attention_mask=True, 
                                      return_token_type_ids=False)
        
        ## deal with mols list
        mol_batch = self.collater([mol for mols in mols_list for mol in mols])
        return sentence_tokens, notes_tokens, mol_batch


class ReactionDataset(InMemoryDataset):
    def __init__(self, root, text_max_len, num_query_token):
        super().__init__(root)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.total_reactions = []
        for file in self.processed_paths[1:]:
            with open(file, 'r') as f:
                reactions = json.load(f)
            self.total_reactions.extend(reactions)
        self.tokenizer = None
        self.text_max_len = text_max_len
        self._graph_collater = Collater([], [])
        self.ph_token = "<mol>" * num_query_token

    def get_mol(self, idx):
        data = Data()
        for key in self._data.keys:
            item, slices = self._data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data
    
    def get(self, idx):
        reaction = self.total_reactions[idx]
        input_mols = reaction['input_mols']
        output_mols = reaction['output_mols']
        notes = reaction['notes']
        input_mols = [(self.get_mol(mol_id), role) for mol_id, role in input_mols]
        output_mols = [(self.get_mol(mol_id), role) for mol_id, role in output_mols]
        
        text = 'The input molecules of a chemical reaction are '
        mols = []
        for i, (mol, role) in enumerate(input_mols):
            if i < len(input_mols) - 1:
                text += f'{role.lower()} {self.ph_token}, ' # use <mol> as a placeholder
            else:
                text += f'{role.lower()} {self.ph_token}. ' # use <mol> as a placeholder
            mols.append(mol)
        text += 'The output molecules are '
        for i, (mol, role) in enumerate(output_mols):
            if i < len(output_mols) - 1:
                text += f' {self.ph_token}, ' # use <mol> as a placeholder
            else:
                text += f' {self.ph_token}. ' # use <mol> as a placeholder
            mols.append(mol)
        # react_tokens = self.tokenizer_text(text) # shape = [B, N]
        # notes_tokens = self.tokenizer_text(notes) # shape = [B, N]
        return text, notes, mols
    
    def __len__(self):
        return len(self.total_reactions)

    def len(self):
        return len(self.total_reactions)
    
    @property
    def raw_file_names(self):
        return [f.name for f in Path(self.raw_dir).glob('*')]

    @property
    def processed_file_names(self):
        return ['data.pt'] + self.raw_file_names

    def process(self):
        # Read data into huge `Data` list.
        mol_set = {}
        for f in tqdm(self.raw_paths):
            filename = Path(f).name
            output_file = Path(self.processed_dir) / filename
            with open(f, 'r') as f:
                reactions = json.load(f)
            processed_reactions = []
            for reaction in reactions:
                input_mols, output_mols, notes = reaction
                if len(input_mols) + len(output_mols) > 8:
                    continue
                
                input_ids = []
                for mol, role in input_mols:
                    if mol in mol_set:
                        mol_id = mol_set[mol]
                    else:
                        mol_set[mol] = len(mol_set)
                        mol_id = len(mol_set) - 1
                    input_ids.append((mol_id, role))
                
                output_ids = []
                for mol, role in output_mols:
                    if mol in mol_set:
                        mol_id = mol_set[mol]
                    else:
                        mol_set[mol] = len(mol_set)
                        mol_id = len(mol_set) - 1
                    output_ids.append((mol_id, role))
                processed_reactions.append({'input_mols': input_ids, 'output_mols': output_ids, 'notes': notes})
            with open(output_file, 'w') as f:
                json.dump(processed_reactions, f, ensure_ascii=False)
        id2mol = {v: smiles2data(k) for k, v in mol_set.items()}
        id2mol = [id2mol[i] for i in range(len(id2mol))]
        data, slices = self.collate(id2mol)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == '__main__':
    print(len(ReactionDataset('../data/ORDCaptioning/train', 128, 8)))
    print(len(ReactionDataset('../data/ORDCaptioning/valid', 128, 8)))
    print(len(ReactionDataset('../data/ORDCaptioning/test', 128, 8)))