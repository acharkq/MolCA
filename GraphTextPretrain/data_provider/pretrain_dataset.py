import torch
from torch_geometric.data import Data, Dataset
import torch_geometric
from utils.GraphAug import drop_nodes, permute_edges, subgraph, mask_nodes
from copy import deepcopy
import numpy as np
import os
import random

class GINPretrainDataset(Dataset):
    def __init__(self, root, text_max_len, graph_aug, declip):
        super(GINPretrainDataset, self).__init__(root)
        self.root = root
        self.graph_aug = graph_aug
        self.text_max_len = text_max_len
        self.graph_name_list = os.listdir(root+'graph/')
        self.graph_name_list.sort()
        self.text_name_list = os.listdir(root+'text/')
        self.text_name_list.sort()
        self.declip = declip
        self.tokenizer = None

    def get(self, index):
        return self.__getitem__(index)

    def len(self):
        return len(self)

    def __len__(self):
        return len(self.graph_name_list)

    def __getitem__(self, index):
        if self.declip:
            graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
            # load and process graph
            graph_path = os.path.join(self.root, 'graph', graph_name)
            data_graph = torch.load(graph_path)
            # try:
            #     data_graph_aug = self.augment(data_graph, self.graph_aug)
            # except:
            #     exit()
            data_graph_aug = self.augment(data_graph, self.graph_aug)
            # load and process text
            text_path = os.path.join(self.root, 'text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                text_list.append(line)
                if count > 100:
                    break
            if len(text_list) < 2:
                two_text_list = [text_list[0], text_list[0]]
            else:
                two_text_list = random.sample(text_list, 2)
            text1, mask1 = self.tokenizer_text(two_text_list[0])
            text2, mask2 = self.tokenizer_text(two_text_list[1])

            return data_graph, data_graph_aug, text1.squeeze(0), mask1.squeeze(0), text2.squeeze(0), mask2.squeeze(0)
        else:
            graph_name, text_name = self.graph_name_list[index], self.text_name_list[index]
            # load and process graph
            graph_path = os.path.join(self.root, 'graph', graph_name)
            data_graph = torch.load(graph_path)
            # load and process text
            text_path = os.path.join(self.root, 'text', text_name)
            text_list = []
            count = 0
            for line in open(text_path, 'r', encoding='utf-8'):
                count += 1
                text_list.append(line)
                if count > 100:
                    break
            text_sample = random.sample(text_list, 1)
            text_list.clear()
            text, mask = self.tokenizer_text(text_sample[0])
            return data_graph, text.squeeze(0), mask.squeeze(0)


    def augment(self, data, graph_aug):
        if graph_aug == 'dnodes':
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
                                        add_special_tokens=True,
                                        max_length=self.text_max_len,
                                        return_tensors='pt',
                                        return_attention_mask=True)
        input_ids = sentence_token['input_ids']
        attention_mask = sentence_token['attention_mask']
        return input_ids, attention_mask

