# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset import MoleculeCaption
from data_provider.reaction_caption_dataset import ReactionDataset, ReactionCollater

class TrainCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)

        ## deal with prompt
        prompt_tokens = self.tokenizer(smiles_prompt, return_tensors='pt', max_length=self.text_max_len, padding='longest', truncation=True, return_attention_mask=True)
        prompt_lens = prompt_tokens.attention_mask.sum(dim=1)

        ## concate text and prompt
        texts = [prompt + text for prompt, text in zip(smiles_prompt, texts)]


        graphs = self.collater(graphs)
        text_tokens = self.tokenizer(text=texts,
                                truncation=True,
                                padding='longest',
                                add_special_tokens=True,
                                max_length=self.text_max_len,
                                return_tensors='pt',
                                return_attention_mask=True)
        return graphs, text_tokens, prompt_lens


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        
    def __call__(self, batch):
        graphs, texts, smiles_prompt = zip(*batch)

        ## deal with prompt
        prompt_tokens = self.tokenizer(smiles_prompt, return_tensors='pt', max_length=self.text_max_len, padding='longest', truncation=True, return_attention_mask=True)

        graphs = self.collater(graphs)
        return graphs, prompt_tokens, texts
    

class Stage2ReactionDM(LightningDataModule):
    def __init__(
        self,
        molecule_root: str = 'data/',
        reaction_root: str = '',
        num_workers: int = 0,
        batch_size: int = 256,
        text_max_len: int = 128,
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        self.mol_pretrain_dataset = MoleculeCaption(molecule_root+f'/pretrain/', text_max_len, self.prompt)
        self.mol_train_dataset = MoleculeCaption(molecule_root+f'/train/', text_max_len, self.prompt)
        self.mol_val_dataset = MoleculeCaption(molecule_root + '/valid/', text_max_len, self.prompt)
        self.mol_test_dataset = MoleculeCaption(molecule_root + '/test/', text_max_len, self.prompt)
        self.reaction_root = reaction_root
        if reaction_root:
            self.rea_train_dataset = ReactionDataset(reaction_root+'/train/', text_max_len, args.num_query_token)
            self.rea_val_dataset = ReactionDataset(reaction_root+'/valid/', text_max_len, args.num_query_token)
            self.rea_test_dataset = ReactionDataset(reaction_root+'/test/', text_max_len, args.num_query_token)
            
        self.init_tokenizer(tokenizer)
    
    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.mol_pretrain_dataset.tokenizer = tokenizer
        self.mol_train_dataset.tokenizer = tokenizer
        self.mol_val_dataset.tokenizer = tokenizer
        self.mol_test_dataset.tokenizer = tokenizer
        if self.reaction_root:
            tokenizer.add_tokens('<mol>')
            tokenizer.mol_token_id = tokenizer("<mol>", add_special_tokens=False).input_ids[0]
            self.mol_token_id = tokenizer.mol_token_id
            self.rea_train_dataset.tokenizer = tokenizer
            self.rea_val_dataset.tokenizer = tokenizer
            self.rea_test_dataset.tokenizer = tokenizer

    def train_dataloader(self):
        mol_loader = DataLoader(
                self.mol_pretrain_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )
        if self.reaction_root:
            rea_loader = DataLoader(
                    self.rea_train_dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=True,
                    persistent_workers=True,
                    collate_fn=ReactionCollater(self.tokenizer, self.text_max_len, self.mol_token_id),
            )
            return [mol_loader, rea_loader]
        return mol_loader

    def val_dataloader(self):
        mol_loader = DataLoader(
                self.mol_val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=True,
                collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )
        test_loader = DataLoader(
            self.mol_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len),
        )
        if self.reaction_root:
            rea_loader = DataLoader(
                    self.rea_val_dataset,
                    batch_size=self.batch_size,
                    shuffle=False,
                    num_workers=self.num_workers,
                    pin_memory=False,
                    drop_last=False,
                    persistent_workers=True,
                    collate_fn=ReactionCollater(self.tokenizer, self.text_max_len, self.mol_token_id),
            )
            return [mol_loader, test_loader, rea_loader]
        return [mol_loader, test_loader]
    
    def test_dataloader(self):
        loader = DataLoader(
            self.mol_test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--inference_batch_size', type=int, default=4)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--molecule_root', type=str, default='data/PubChemDataset_v4')
        parser.add_argument('--reaction_root', type=str, default='data/ORDCaptioning')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_SMILES]{}[END_SMILES]. ')
        return parent_parser
    