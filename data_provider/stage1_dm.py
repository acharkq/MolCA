# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
from pytorch_lightning import LightningDataModule
import torch_geometric
from torch_geometric.data import Batch
from data_provider.pretrain_dataset import GINPretrainDataset
from data_provider.retrieval_dataset import RetrievalDataset
from data_provider.molecule_caption_dataset import MoleculeCaptionV2
from torch.utils.data import DataLoader


class TrainCollater(object):
    def __init__(self, tokenizer, text_max_len):
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
    
    def __call__(self, batch):
        data_list, text_list, smiles_prompt_list = zip(*batch)
        graph_batch = Batch.from_data_list(data_list)        
        text_batch = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
        return graph_batch, text_batch.input_ids, text_batch.attention_mask

# class TrainCollater(object):
#     def __init__(self, tokenizer, text_max_len):
#         self.tokenizer = tokenizer
#         self.text_max_len = text_max_len
    
#     def __call__(self, batch):
#         data_list, text_list, smiles_prompt_list = zip(*batch)
#         graph_batch = Batch.from_data_list(data_list[0], data_list)        
#         text_batch = self.tokenizer(text_list, padding='max_length', truncation=True, max_length=self.text_max_len, return_tensors='pt')
#         return graph_batch, text_batch.input_ids, text_batch.attention_mask


class Stage1DM(LightningDataModule):
    def __init__(
        self,
        num_workers: int = 0,
        batch_size: int = 256,
        root: str = 'data/',
        text_max_len: int = 128,
        graph_aug: str = 'dnodes',
        tokenizer=None,
        args=None,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.match_batch_size = args.match_batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer
        self.text_max_len = text_max_len
        print('Loading PubChem324k dataset')
        if False:
            self.train_dataset = GINPretrainDataset(root+'/pretrain/', text_max_len, graph_aug, args.text_aug, args.filtered_cid_path)
            self.val_dataset = GINPretrainDataset(root + '/valid/', text_max_len, graph_aug, args.text_aug)
            self.val_dataset_match = RetrievalDataset(root + '/valid/', args).shuffle()
            self.test_dataset_match = RetrievalDataset(root + '/test/', args).shuffle()
            self.val_match_loader = torch_geometric.loader.DataLoader(self.val_dataset_match, 
                                                                  batch_size=self.match_batch_size,
                                                                  shuffle=False,
                                                                  num_workers=self.num_workers, 
                                                                  pin_memory=False, 
                                                                  drop_last=False, 
                                                                  persistent_workers=True)
            self.test_match_loader = torch_geometric.loader.DataLoader(self.test_dataset_match, 
                                                                    batch_size=self.match_batch_size,
                                                                    shuffle=False,
                                                                    num_workers=self.num_workers, 
                                                                    pin_memory=False, 
                                                                    drop_last=False, 
                                                                    persistent_workers=True)
        else:
            self.train_dataset = MoleculeCaptionV2(root+'pretrain.pt', text_max_len)
            self.val_dataset = MoleculeCaptionV2(root + 'valid.pt', text_max_len)
            self.val_dataset_match = MoleculeCaptionV2(root + 'valid.pt', text_max_len).shuffle()
            self.test_dataset_match = MoleculeCaptionV2(root + 'test.pt', text_max_len).shuffle()
            self.val_match_loader = DataLoader(self.val_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
            self.test_match_loader = DataLoader(self.test_dataset_match, batch_size=self.match_batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, text_max_len))
    
    def train_dataloader(self):
        if False:
            loader = torch_geometric.loader.DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=True,
                persistent_workers=True
            )
        else:
            loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, pin_memory=False, drop_last=True, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        # print('len(train_dataloader)', len(loader))
        return loader

    def val_dataloader(self):
        if False:
            loader = torch_geometric.loader.DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=False,
                drop_last=False,
                persistent_workers=True
            )
        else:
            loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, pin_memory=False, drop_last=False, persistent_workers=True, collate_fn=TrainCollater(self.tokenizer, self.text_max_len))
        return loader

    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=4)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--match_batch_size', type=int, default=64)
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/PubChemDataset/PubChem-320k')
        parser.add_argument('--text_max_len', type=int, default=128)
        parser.add_argument('--graph_aug', type=str, default='dnodes')
        parser.add_argument('--text_aug', action='store_true', default=False)
        parser.add_argument('--use_phy_eval', action='store_true', default=False)
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        return parent_parser
    