import torch
from torch_geometric.data import Dataset

from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random
from transformers import BertTokenizer

class GINMatchDataset(Dataset):
    def __init__(self, root, args):
        super(GINMatchDataset, self).__init__(root)
        self.root = root
        self.graph_aug = args.graph_aug
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = BertTokenizer.from_pretrained('../GraphTextPretrain/bert_pretrained/')

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    # def __getitem__(self, index):
    #     graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
    #     graph_path = os.path.join(self.root, 'graph', graph_name)
    #     data_graph = torch.load(graph_path)
    #
    #     data_aug = self.augment(data_graph, self.graph_aug)
    #     text_path = os.path.join(self.root, 'text', text_name)
    #
    #     text_list = []
    #     count = 0
    #     for line in open(text_path, 'r', encoding='utf-8'):
    #         count += 1
    #         line.strip('\n')
    #         text_list.append(line)
    #         if count > 500:
    #             break
    #
    #     # para-level
    #     text, mask = self.tokenizer_text(text_list[0][:256])
    #     return data_aug, text.squeeze(0), mask.squeeze(0)  # , index

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        assert graph_name.strip('.pt').strip('graph_') == text_name.strip('.txt').strip('text_') == smiles_name.strip('.txt').strip('smiles_'), print(graph_name, text_name, smiles_name)
        

        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)
        data_aug = self.augment(data_graph, self.graph_aug)

        text_path = os.path.join(self.root, 'smiles', smiles_name)
        text = 'This molecule is '
        count = 0
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text += line
            if count > 1:
                break

        text_path = os.path.join(self.root, 'text', text_name)
        count = 0
        text += '. '
        for line in open(text_path, 'r', encoding='utf-8'):
            count += 1
            line.strip('\n')
            text += line
            if count > 100:
                break

        # para-level
        text, mask = self.tokenizer_text(text)

        # random sentence
        # if self.data_type == 1:
        #     sts = text_list[0].split('.')
        #     remove_list = []
        #     for st in (sts):
        #         if len(st.split(' ')) < 5:
        #             remove_list.append(st)
        #     remove_list = sorted(remove_list, key=len, reverse=False)
        #     for r in remove_list:
        #         if len(sts) > 1:
        #             sts.remove(r)
        #     text_index = random.randint(0, len(sts) - 1)
        #     text, mask = self.tokenizer_text(sts[text_index])

        return data_aug, text.squeeze(0), mask.squeeze(0)  # , index

    def augment(self, data, graph_aug):

        if graph_aug == 'noaug':
            data_aug = deepcopy(data)
        elif graph_aug == 'dnodes':
            data_aug = drop_nodes(deepcopy(data))
        elif graph_aug == 'pedges':
            data_aug = permute_edges(deepcopy(data))
        elif graph_aug == 'subgraph':
            data_aug = subgraph(deepcopy(data))
        elif graph_aug == 'mask_nodes':
            data_aug = mask_nodes(deepcopy(data))
        elif graph_aug == 'random2':  # choose one from two augmentations
            n = np.random.randint(2)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random3':  # choose one from three augmentations
            n = np.random.randint(3)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            else:
                print('sample error')
                assert False
        elif graph_aug == 'random4':  # choose one from four augmentations
            n = np.random.randint(4)
            if n == 0:
                data_aug = drop_nodes(deepcopy(data))
            elif n == 1:
                data_aug = permute_edges(deepcopy(data))
            elif n == 2:
                data_aug = subgraph(deepcopy(data))
            elif n == 3:
                data_aug = mask_nodes(deepcopy(data))
            else:
                print('sample error')
                assert False
        else:
            data_aug = deepcopy(data)
            data_aug.x = torch.ones((data.edge_index.max()+1, 1))

        return data_aug

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=False,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask


class CapDataset(Dataset):
    def __init__(self, root, args):
        super(CapDataset, self).__init__(root)
        self.root = root
        self.text_max_len = args.text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.smiles_name_list = os.listdir(root + 'smiles/')
        self.smiles_name_list.sort()
        self.tokenizer = self.extend_tokenizer()

    def extend_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        tokenizer.add_special_tokens({'bos_token': '[DEC]'})
        tokenizer.add_special_tokens({'additional_special_tokens': ['[ENC]']})
        tokenizer.enc_token_id = tokenizer.additional_special_tokens_ids[0]
        return tokenizer

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        graph_name, text_name, smiles_name = self.graph_name_list[index], self.text_name_list[index], self.smiles_name_list[index]
        graph_path = os.path.join(self.root, 'graph', graph_name)
        data_graph = torch.load(graph_path)

        text_path = os.path.join(self.root, 'text', text_name)
        text = ''
        for line in open(text_path, 'r', encoding='utf-8'):
            text += line

        smiles_path = os.path.join(self.root, 'smiles', smiles_name)
        raw_smiles = ''
        for line in open(smiles_path, 'r', encoding='utf-8'):
            raw_smiles += line
        raw_smiles += 'is\t'
        smiles, smiles_mask = self.tokenizer_text(raw_smiles)
        return data_graph, smiles.squeeze(0), smiles_mask.squeeze(0), text
        # text, mask = self.tokenizer_text(text[:self.text_max_len])
        # return data_graph, text.squeeze(0), mask.squeeze(0)

    def tokenizer_text(self, text):
        sentence_token = self.tokenizer(text=text,
                                        truncation=True,
                                        padding='max_length',
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

    # def tokenizer_text(self, text):
    #     sentence_token = self.tokenizer(text=text + ' is ',
    #                                     return_tensors='pt',
    #                                     padding='max_length',
    #                                     max_length=self.text_max_len,
    #                                     return_attention_mask=True)
    #     input_ids = sentence_token['input_ids']
    #     attention_mask = sentence_token['attention_mask']
    #     return input_ids, attention_mask
