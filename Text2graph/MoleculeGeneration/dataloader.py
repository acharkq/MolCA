# from ogb.utils import smiles2graph
import torch_geometric
from torch_geometric.data import Data, Dataset
import torch
from transformers import T5Tokenizer
import os.path as osp

import csv
import pickle

import spacy


class TextMoleculeReplaceDataset(Dataset): #, tokenizer#This dataset replaces the name of the molecule at the beginning of the description
    def __init__(self, data_path, split):
        self.data_path = data_path
        
        # self.tokenizer = tokenizer

        self.cids = []
        self.descriptions = {}
    
        self.cids_to_smiles = {}
        self.smiles = {}
        
        #load data
        with open(osp.join(data_path, split+'.txt')) as f:
            reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            for n, line in enumerate(reader):
                # print(line)
                # print(line['CID'])
                self.descriptions[line['CID']] = line['description']
                self.cids_to_smiles[line['CID']] = line['SMILES']
                self.cids.append(line['CID'])


    def __len__(self):
        return len(self.cids)


    def __getitem__(self, idx):

        cid = self.cids[idx]

        smiles = self.cids_to_smiles[cid]

        description = self.descriptions[cid]

        # ori_graph = smiles2graph(smiles)
        # x = torch.from_numpy(ori_graph['node_feat']).to(torch.int64)
        # # print(x)
        # edge_index = torch.from_numpy(ori_graph['edge_index']).to(torch.int64)
        # edge_attr = torch.from_numpy(ori_graph['edge_feat']).to(torch.int64)
        # num_nodes = int(ori_graph['num_nodes'])
        # graph = Data(x, edge_index, edge_attr, num_nodes=num_nodes)

        # text = self.tokenizer(description, padding="max_length", max_length=512, truncation=True, return_tensors='pt')

        # # smiles_tokens = self.smiles_tokenizer.get_tensor(smiles)
        # smiles_tokens = self.tokenizer(smiles, padding="max_length", max_length=512, truncation=True, return_tensors='pt')

        # return {'graph': graph, 'smiles':smiles, 'description':description, 
        # 'smiles_tokens':smiles_tokens['input_ids'].squeeze(), 'smiles_mask':smiles_tokens['attention_mask'].squeeze(),
        # 'text':text['input_ids'].squeeze(), 'text_mask':text['attention_mask'].squeeze()}

        return {'smiles':smiles, 'description':description}


if __name__ == '__main__':
    tokenizer = T5Tokenizer.from_pretrained("molt5_base/", model_max_length=512)
    mydataset = TextMoleculeReplaceDataset('data/', 'test', tokenizer)
    train_loader = torch_geometric.loader.DataLoader(
            mydataset,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=True,
            # persistent_workers = True
        )
    for i, a in enumerate(train_loader):
        print(a)
        text_mask = a['text_mask']
        label = a['text']  # caption本身
        label = label.masked_fill(~text_mask.bool(), -100)
        print(label)
        break
