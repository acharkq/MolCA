# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torch_geometric.data.dataset import Dataset
from torch_geometric.loader.dataloader import Collater
import pandas as pd
import re
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from torch_geometric.data import Dataset
from data_provider.loader import MoleculeDataset
from data_provider.splitters import scaffold_split

# we split individual characters inside special tokens like [START_DNA]
CUSTOM_SEQ_RE = re.compile(r"(\[START_(DNA|SMILES|I_SMILES|AMINO)])(.*?)(\[END_\2])")

# token added to implement a custom sequence tokenization. This token is added at
# corpus cleaning step and removed in pretokenization. The digits are added to increase the chance
# that they do not occur in the corpus. The digits are escaped so that the token does not appear
# literally in the source code in case we ever include it in the training data.
SPLIT_MARKER = f"SPL{1}T-TH{1}S-Pl3A5E"

def _insert_split_marker(m: re.Match):
    """
    Applies split marker based on a regex match of special tokens such as
    [START_DNA].

    Parameters
    ----------
    n : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    start_token, _, sequence, end_token = m.groups()
    sequence = re.sub(r"(.)", fr"{SPLIT_MARKER}\1", sequence, flags=re.DOTALL)
    return f"{start_token}{sequence}{SPLIT_MARKER}{end_token}"


def smiles_handler(text, mol_ph):
    smiles_list = []
    for match in CUSTOM_SEQ_RE.finditer(text):
        smiles = match.group(3)
        smiles_list.append(smiles)
    
    text = CUSTOM_SEQ_RE.sub(r'\1\3\4%s' % (mol_ph), text)
    text = escape_custom_split_sequence(text)
    return text, smiles_list


def escape_custom_split_sequence(text):
    """
    Applies custom splitting to the text for GALILEO's tokenization

    Parameters
    ----------
    text : str
        Input text to split

    Returns
    ----------
    str - the text with the split token added
    """
    return CUSTOM_SEQ_RE.sub(_insert_split_marker, text)

class MPPCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id, num_labels):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        self.num_labels = num_labels
        
    def __call__(self, batch):
        graphs, smiles_prompt = zip(*batch)
        graphs = self.collater(graphs)
        
        ## deal with prompt
        if self.mol_token_id:
            smiles_prompt = [smiles_handler(p, self.mol_ph)[0] for p in smiles_prompt]
            smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                                  truncation=False,
                                                  padding='longest',
                                                  add_special_tokens=True,
                                                  return_tensors='pt',
                                                  return_attention_mask=True)

            is_mol_token = smiles_prompt_tokens.input_ids == self.mol_token_id
            smiles_prompt_tokens['is_mol_token'] = is_mol_token
        else:
            smiles_prompt = [escape_custom_split_sequence(p) for p in smiles_prompt]
            smiles_prompt_tokens = self.tokenizer(text=smiles_prompt, 
                                                  truncation=False,
                                                  padding='longest',
                                                  add_special_tokens=True,
                                                  return_tensors='pt',
                                                  return_attention_mask=True)
        if len(graphs.y.shape) == 1:
            graphs['y'] = graphs.y.reshape(-1, self.num_labels)
        return graphs, smiles_prompt_tokens


def load_dataset_chem(root, tuning_dataset, split):
    dataset = MoleculeDataset(
        root + tuning_dataset, dataset=tuning_dataset)
    print(dataset)
    if split == "scaffold":
        smiles_list = pd.read_csv(
            root + tuning_dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_dataset, valid_dataset, test_dataset, (train_smiles, valid_smiles, test_smiles) = scaffold_split(
            dataset, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1, return_smiles=True)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")
    return train_dataset, valid_dataset, test_dataset, train_smiles, valid_smiles, test_smiles


class MPPSmilesDataset(Dataset):
    def __init__(self, g_dataset, smiles_list, prompt):
        assert len(g_dataset) == len(smiles_list)
        self.g_dataset = g_dataset
        self.smiles_list = smiles_list
        self.prompt = prompt
        # self.num_labels = self.get(0)[0].y.shape[-1]
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        assert self.prompt.find('{}') >= 0
        smiles_prompt = self.prompt.format(self.smiles_list[idx][:256])
        return self.g_dataset[idx], smiles_prompt
    
    def len(self):
        return len(self)

    def get(self, idx):
        return self.__getitem__(idx)

class MPPDM(LightningDataModule):
    def __init__(
        self,
        mode: str = 'pretrain',
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt

        train_dataset, valid_dataset, test_dataset, train_smiles, valid_smiles, test_smiles = \
            load_dataset_chem(args.root, args.tuning_dataset, 'scaffold')
        self.train_dataset = MPPSmilesDataset(train_dataset, train_smiles, self.prompt)
        self.val_dataset = MPPSmilesDataset(valid_dataset, valid_smiles, self.prompt)
        self.test_dataset = MPPSmilesDataset(test_dataset, test_smiles, self.prompt)
        self.init_tokenizer(tokenizer)
        self.mol_ph_token = '<mol>' * self.args.num_query_token
        self.mol_token_id = None

    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer
        try:
            self.mol_token_id = self.tokenizer.mol_token_id
            print('load tokenizer success')
        except AttributeError:
            print('load tokenizer failed')
        # self.tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]

    def train_dataloader(self):
        assert self.mode == 'ft'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MPPCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.args.num_labels),
        )
        return loader
    
    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=MPPCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.args.num_labels),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=MPPCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id, self.args.num_labels),
        )
        return [val_loader, test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=128)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/MoleculeNet/')
        parser.add_argument('--tuning_dataset', type=str, default='')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    