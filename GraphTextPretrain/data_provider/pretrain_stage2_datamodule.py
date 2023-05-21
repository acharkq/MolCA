# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch.utils.data import DataLoader
from torch_geometric.loader.dataloader import Collater
from data_provider.molecule_caption_dataset import MoleculeCaption


class MyCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])
        
    def __call__(self, batch):
        graphs, texts = zip(*batch)
        batch_graph = self.collater(graphs)
        tokens = self.tokenizer(text=texts,
                                truncation=True,
                                padding='longest',
                                add_special_tokens=True,
                                max_length=self.text_max_len,
                                return_tensors='pt',
                                return_attention_mask=True)
        input_ids = tokens['input_ids']
        attention_mask = tokens['attention_mask']
        return batch_graph, input_ids, attention_mask
    
    
class PretrainStage2DataModule(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.train_dataset = MoleculeCaption(root+'/train/', text_max_len)
        self.val_dataset = MoleculeCaption(root + '/valid/', text_max_len)
        self.tokenizer = None
    
    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.text_max_len),
        )
        # loader = torch_geometric.loader.DataLoader(
        #     self.train_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=True,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     drop_last=True,
        #     persistent_workers=True
        # )
        return loader

    def val_dataloader(self):
        # loader = torch_geometric.loader.DataLoader(
        #     self.val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     pin_memory=False,
        #     drop_last=False,
        #     persistent_workers=True
        # )
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=MyCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset/PubChem-320k')
        parser.add_argument('--text_max_len', type=int, default=128)
        return parent_parser
    