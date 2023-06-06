import os
import pandas as pd
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_opt import Blip2OPT
from model.blip2_stage2 import Blip2Stage2
# from data_provider.stage2_dm import Stage2DM
from data_provider.splitters import scaffold_split_without_dataset
from data_provider.loader import MoleculeDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import random

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)



def load_dataset_chem(mpp_dataset, split):
    dataset = MoleculeDataset("data/MoleculeNet/" + mpp_dataset, dataset=mpp_dataset)
    print(dataset)
    if split == "scaffold":
        smiles_list = pd.read_csv(
            'data/MoleculeNet/' + mpp_dataset + '/processed/smiles.csv', header=None)[0].tolist()
        train_smiles, valid_smiles, test_smiles, train_y, valid_y, test_y = scaffold_split_without_dataset(dataset.data.y, smiles_list, null_value=0, frac_train=0.8, frac_valid=0.1, frac_test=0.1)
        print("scaffold")
    else:
        raise ValueError("Invalid split option.")
    return train_smiles, valid_smiles, test_smiles, train_y, valid_y, test_y


class PromptDataset(Dataset):
    def __init__(self, dataset, num_shots=5, split='scaffold'):
        train_smiles, valid_smiles, test_smiles, train_y, valid_y, test_y = \
            load_dataset_chem(dataset, split)
        self.train_smiles = train_smiles
        self.valid_smiles = valid_smiles
        self.test_smiles = test_smiles
        if len(train_y.shape) == 1:
            self.train_y = train_y.reshape(-1, 1)
            self.valid_y = valid_y.reshape(-1, 1)
            self.test_y = test_y.reshape(-1, 1)
        
        ## construct in context learning prompt

        if dataset.lower() == 'bbbp':
            prompt_func = self.bbbp_prompt
        elif dataset.lower() == 'bace':
            prompt_func = self.bace_prompt
        elif dataset.lower() == 'hiv':
            prompt_func = self.hiv_prompt
        elif dataset.lower() == 'clintox':
            prompt_func = self.clintox_prompt
        else:
            raise NotImplementedError()
        self.prompt_func = prompt_func

        self.num_shots = num_shots
        
        self.neg_smiles = []
        self.pos_smiles = []
        print(self.train_y.shape)
        for task_id in range(self.train_y.shape[1]):
            neg_ids = (self.train_y[:, task_id] == -1).nonzero(as_tuple=True)[0]
            pos_ids = (self.train_y[:, task_id] == 1).nonzero(as_tuple=True)[0]
            self.neg_smiles.append([])
            self.pos_smiles.append([])
            for idx in neg_ids:
                self.neg_smiles[-1].append(train_smiles[idx])
            for idx in pos_ids:
                self.pos_smiles[-1].append(train_smiles[idx])

    def bbbp_prompt(self, smiles, task_id=0):
        '''
        galactica version
        '''
        prompt = 'Here is a SMILES formula: [START_I_SMILES]%s[END_I_SMILES]\n Question: Will the chemical compound penetrate the blood-brain barrier?' % smiles
        return prompt

    def bace_prompt(self, smiles, task_id=0):
        prompt = 'Will [START_I_SMILES]%s[END_I_SMILES] bind to human beta-secretase 1 (BACE-1)?' % smiles
        return prompt

    def hiv_prompt(self, smiles, task_id=0):
        prompt = 'Will the compound [START_I_SMILES]%s[END_I_SMILES] inhibit HIV activity?' % smiles
        return prompt

    def clintox_prompt(self, smiles, task_id=0):
        if task_id == 0:
            prompt1 = "Will the compound [START_I_SMILES]%s[END_I_SMILES] be approved by the Food and Drug Administration? Yes or no?" % smiles
            return prompt1
        elif task_id == 1:
            prompt2 = 'Predict whether the compound [START_I_SMILES]%s[END_I_SMILES] will demonstrate clinical trial toxicity. Yes or no?' % smiles
            return prompt2
        else:
            raise NotImplementedError()
    
    # def tox21_prompt(self, smiles, task_id):
    #     prompt = 'Here is a isomeric SMILES for a compound: \n [START_I_SMILES]%s[END_I_SMILES] \n Q: Will the compound be active against the mitochondrial membrane potential?' % ()

    def __len__(self):
        return len(self.test_smiles)
    
    def __getitem__(self, idx):
        icl_list = []
        for task_id in range(self.train_y.shape[1]):
            pos_num = self.num_shots // 2
            neg_num = self.num_shots - pos_num
            pos_smiles = random.sample(self.pos_smiles[task_id], k=pos_num)
            neg_smiles = random.sample(self.neg_smiles[task_id], k=neg_num)
            
            icl = []
            for smiles in pos_smiles:
                prompt = self.prompt_func(smiles, task_id)
                prompt += ' Answer: Yes.'
                icl.append(prompt)
            for smiles in neg_smiles:
                prompt = self.prompt_func(smiles, task_id)
                prompt += ' Answer: No.'
                icl.append(prompt)
            random.shuffle(icl)
            icl = ' '.join(icl)
            prompt = self.prompt_func(self.test_smiles[idx], task_id) + ' Answer: '
            icl = ' '.join([icl, prompt])
            icl_list.append(icl)
        return tuple(icl_list)

        

def main(args):
    pl.seed_everything(args.seed)
    # model
    model= Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.use_bn, args.llm_tune, args.peft_dir, args.opt_model, args.prompt)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')['state_dict']
        state_dict = {k[9:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    
    model.to(f'cuda:{args.device}')
    print('total params:', sum(p.numel() for p in model.parameters()))
    dataset = PromptDataset(args.mpp_dataset, args.num_shots)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False, num_workers=2)
    for data_list in tqdm(dataloader):
        for data in data_list:
            print('----------------')
            print(data)
            samples = {'prompts': data}
            output = model.qa(
                samples, 
                do_sample=True,
                num_beams=5,
                max_length=128,
                min_length=8,
                temperature=0.7,
            )
            print(output)

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--mpp_dataset', type=str, default='bbbp')
    parser.add_argument('--num_shots', type=int, default=5)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    # parser = Stage2DM.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision=16,
                        max_epochs=10,
                        accumulate_grad_batches=1,
                        check_val_every_n_epoch=1)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args

if __name__ == '__main__':
    args = get_args()
    main(args)

