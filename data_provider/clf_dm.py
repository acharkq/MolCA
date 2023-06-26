# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
# from data_provider.molecule_caption_dataset import MoleculeCaption
from data_provider.molecule_fgpred_dataset import MoleculeFGPred
import re
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

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

class ClfCollater:
    def __init__(self, tokenizer, text_max_len, mol_ph, mol_token_id):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.mol_ph = mol_ph
        self.mol_token_id = mol_token_id
        
    def __call__(self, batch):
        graphs, smiles_prompt, labels = zip(*batch)
        labels = torch.stack(labels, dim=0)
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

        graphs['y'] = labels
        return graphs, smiles_prompt_tokens


class FGClfDM(LightningDataModule):
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
        self.train_dataset = MoleculeFGPred(root+f'/train/', text_max_len, self.prompt, args.task_type)
        self.val_dataset = MoleculeFGPred(root + '/valid/', text_max_len, self.prompt, args.task_type)
        self.test_dataset = MoleculeFGPred(root + '/test/', text_max_len, self.prompt, args.task_type)
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
            collate_fn=ClfCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
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
            collate_fn=ClfCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=ClfCollater(self.tokenizer, self.text_max_len, self.mol_ph_token, self.mol_token_id),
        )
        return [val_loader, test_loader]

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=128)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    