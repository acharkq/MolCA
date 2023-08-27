# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import re
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
# from data_provider.molecule_caption_dataset import MoleculeCaption
from data_provider.smiles_caption_dataset import SmilesCaption


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



class TrainCollater:
    def __init__(self, tokenizer, text_max_len, use_gal):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.use_gal = use_gal
        
    def __call__(self, batch):
        smiles_prompt, texts = zip(*batch)
        if self.use_gal:
            smiles_prompt = [escape_custom_split_sequence(p) for p in smiles_prompt]
        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt,
                                              truncation=True,
                                              padding='longest',
                                              add_special_tokens=True,
                                              max_length=self.text_max_len,
                                              return_tensors='pt',
                                              return_attention_mask=True)
        text_tokens = self.tokenizer(text=texts,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        
        return smiles_prompt_tokens, text_tokens


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len, use_gal):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        self.use_gal = use_gal
        
    def __call__(self, batch):
        smiles_prompt, texts = zip(*batch)
        if self.use_gal:
            smiles_prompt = [escape_custom_split_sequence(p) for p in smiles_prompt]
        ## deal with prompt
        smiles_prompt_tokens = self.tokenizer(text=smiles_prompt,
                                              truncation=True,
                                              padding='longest',
                                              add_special_tokens=True,
                                              max_length=self.text_max_len,
                                              return_tensors='pt',
                                              return_attention_mask=True)
        return smiles_prompt_tokens, texts



class SmilesCaptionDM(LightningDataModule):
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
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        if root.lower().find('chebi') >= 0:
            self.train_dataset = CheBIDataset(root+f'/train.txt', text_max_len, self.prompt)
            self.val_dataset = CheBIDataset(root + '/validation.txt', text_max_len, self.prompt)
            self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len, self.prompt)
        else:
            self.pretrain_dataset = SmilesCaption(root+f'/pretrain/', text_max_len, self.prompt)
            self.train_dataset = SmilesCaption(root+f'/train/', text_max_len, self.prompt)
            self.val_dataset = SmilesCaption(root + '/valid/', text_max_len, self.prompt)
            self.test_dataset = SmilesCaption(root + '/test/', text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)
        self.use_gal = args.llm_name.find('gal') >= 0
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        if hasattr(self, 'pretrain_dataset'):
            self.pretrain_dataset.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer

    def train_dataloader(self):
        if self.mode == 'pretrain':
            loader = DataLoader(
                self.pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.use_gal),
            )
        elif self.mode == 'ft':
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.use_gal),
            )
        else:
            raise NotImplementedError
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
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len, self.use_gal),
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.use_gal),
        )
        return [val_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len, self.use_gal),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=2)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
        return parent_parser
    

class CheBIDataset(Dataset):
    def __init__(self, path, text_max_len, prompt=None):
        self.path = path
        self.text_max_len = text_max_len
        self.prompt = prompt

        if not prompt:
            self.prompt = 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. '
        else:
            self.prompt = prompt

        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines][1:]
        
        self.smiles_list = []
        self.text_list = []
        for line in lines:
            _, smiles, text = line.split('\t')
            self.smiles_list.append(smiles)
            self.text_list.append(text)

    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        text = self.text_list[index] + '\n'

        if self.prompt.find('{}') >= 0:
            smiles_prompt = self.prompt.format(smiles[:128])
        else:
            smiles_prompt = self.prompt
        return smiles_prompt, text