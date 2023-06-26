import os
import json
import pandas as pd
import torch
import argparse
import warnings
import numpy as np
import pytorch_lightning as pl
from model.blip2_opt import Blip2OPT
from model.blip2_stage2 import Blip2Stage2
from pathlib import Path
from data_provider.splitters import scaffold_split_without_dataset
from data_provider.loader import MoleculeDataset
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
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
        self.train_y = train_y.numpy()
        self.valid_y = valid_y.numpy()
        self.test_y = test_y.numpy()
        # print(self.train_y)
        # print(self.valid_y)
        # print(self.test_y)
        if len(train_y.shape) == 1:
            self.train_y = self.train_y.reshape(-1, 1)
            self.valid_y = self.valid_y.reshape(-1, 1)
            self.test_y = self.test_y.reshape(-1, 1)
        
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
        for task_id in range(self.train_y.shape[1]):
            # neg_ids = (self.train_y[:, task_id] == -1).nonzero(as_tuple=True)[0]
            neg_ids = (self.train_y[:, task_id] == -1).nonzero()[0]
            pos_ids = (self.train_y[:, task_id] == 1).nonzero()[0]
            self.neg_smiles.append([])
            self.pos_smiles.append([])
            for idx in neg_ids:
                self.neg_smiles[-1].append(train_smiles[idx])
            for idx in pos_ids:
                self.pos_smiles[-1].append(train_smiles[idx])

    def bbbp_prompt(self, smiles, task_id=0, label='True'):
        '''
        galactica version
        '''
        # prompt = 'Here is a SMILES formula: [START_I_SMILES]%s[END_I_SMILES]\n Question: Will the chemical compound penetrate the blood-brain barrier?' % smiles
        prompt = '[START_I_SMILES]%s[END_I_SMILES]. Question: Will this chemical compound penetrate the blood-brain barrier, Yes or No?' % smiles
        # if label == 'True':
        #     prompt = '[START_I_SMILES]%s[END_I_SMILES]. This chemical compound can penetrate the blood-brain barrier.' % smiles
        # elif label == 'False':
        #     prompt = '[START_I_SMILES]%s[END_I_SMILES]. This chemical compound cannot penetrate the blood-brain barrier.' % smiles
        # elif label == 'Unknown':
        #     prompt = '[START_I_SMILES]%s[END_I_SMILES]. This chemical compound' % smiles
        # else:
        #     raise NotImplementedError
        if label == 'True':
            prompt = '[START_I_SMILES]%s[END_I_SMILES]. Question: Will this chemical compound penetrate the blood-brain barrier, Yes or No? Answer: Yes. ' % smiles
        elif label == 'False':
            prompt = '[START_I_SMILES]%s[END_I_SMILES]. Question: Will this chemical compound penetrate the blood-brain barrier, Yes or No? Answer: No. ' % smiles
        elif label == 'Unknown':
            prompt = '[START_I_SMILES]%s[END_I_SMILES]. Question: Will this chemical compound penetrate the blood-brain barrier, Yes or No? Answer: ' % smiles
        else:
            raise NotImplementedError
        return prompt

    @classmethod
    def bbbp_parse_label(self, line):
        if line.lower().find('no') >= 0:
            return -1
        return 1

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
            neg_num = self.num_shots // 2
            pos_num = self.num_shots - neg_num
            pos_smiles = random.sample(self.pos_smiles[task_id], k=pos_num)
            neg_smiles = random.sample(self.neg_smiles[task_id], k=neg_num)
            
            icl = []
            for smiles in pos_smiles:
                prompt = self.prompt_func(smiles, task_id, 'True')
                icl.append(prompt)
            for smiles in neg_smiles:
                prompt = self.prompt_func(smiles, task_id, 'False')
                icl.append(prompt)
            random.shuffle(icl)
            icl = ' '.join(icl)
            prompt = self.prompt_func(self.test_smiles[idx], task_id, 'Unknown')
            icl = ' '.join([icl, prompt])
            icl_list.append(icl)
        return tuple(icl_list)


def evaluation(y_true, y_scores):
    roc_list = []
    for i in range(y_true.shape[1]):
        # AUC is only defined when there is at least one positive data.
        if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == -1) > 0:
            is_valid = y_true[:, i]**2 > 0
            roc_list.append(roc_auc_score(
                (y_true[is_valid, i] + 1)/2, y_scores[is_valid, i]))

    if len(roc_list) < y_true.shape[1]:
        print("Some target is missing!")
        print("Missing ratio: %f" % (1 - float(len(roc_list))/y_true.shape[1]))
    mean_roc = sum(roc_list) / len(roc_list)
    return mean_roc

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
    model.eval()
    print('total params:', sum(p.numel() for p in model.parameters()))
    dataset = PromptDataset(args.mpp_dataset, args.num_shots)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=2)
    
    num_task_dict = {'tox21': 12, 'hiv': 1, 'muv': 17, 'bace': 1,
                         'bbbp': 1, 'toxcast': 617, 'sider': 27, 'clintox': 2}

    total_predictions = [[] for _ in range(num_task_dict[args.mpp_dataset])]
    output_list = [[] for _ in range(num_task_dict[args.mpp_dataset])]
    output_scores = True
    if output_scores:
        cand_ids = torch.cat([model.opt_tokenizer('Yes', return_tensors='pt').input_ids, model.opt_tokenizer('No', return_tensors='pt').input_ids], dim=0)
        cand_ids = cand_ids.squeeze(-1)
        for task_list in tqdm(dataloader):
            for i, data in enumerate(task_list):
                samples = {'prompts': data}
                if args.mode == 'blip':
                    output = model.blip_qa(
                        samples, 
                        do_sample=False,
                        num_beams=1,
                        max_length=10,
                        min_length=1,
                        temperature=0.7,
                        output_scores=True,
                        top_p=1,
                    )
                elif args.mode == 'opt':
                    output = model.opt_qa(
                        samples, 
                        do_sample=False,
                        num_beams=1,
                        max_length=10,
                        min_length=1,
                        temperature=0.7,
                        output_scores=True,
                        top_p=1,
                    )
                else:
                    raise NotImplementedError()
                predictions = sum(output['scores'][:3])[:, cand_ids] # shape = [B, 2]
                predictions = predictions.softmax(dim=-1)[:, 0] # shape = [B]
                total_predictions[i].extend(predictions.tolist())
                output_list[i].extend(output)
    else:
        for task_list in tqdm(dataloader):
            for i, data in enumerate(task_list):
                samples = {'prompts': data}
                if args.mode == 'blip':
                    output = model.blip_qa(
                        samples, 
                        do_sample=True,
                        num_beams=1,
                        max_length=10,
                        min_length=1,
                        temperature=0.7,
                    )
                elif args.mode == 'opt':
                    output = model.opt_qa(
                        samples, 
                        do_sample=True,
                        num_beams=1,
                        max_length=10,
                        min_length=1,
                        temperature=0.7,
                    )
                else:
                    raise NotImplementedError()
                predictions = [PromptDataset.bbbp_parse_label(item) for item in output]
                total_predictions[i].extend(predictions)
                output_list[i].extend(output)
    total_predictions = np.asarray(total_predictions).T
    log_dir = Path(f'all_checkpoints/{args.filename}')
    log_dir.mkdir(exist_ok=True)
    
    mean_roc = evaluation(dataset.test_y, total_predictions)
    mean_roc = round(mean_roc * 100, 2)
    print('mean roc', mean_roc)

    with open(log_dir / 'predictions.json', 'w', encoding='utf-8') as f:
        json.dump(output_list, f, ensure_ascii=True)

    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--prompt', type=str, default='')
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='blip')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--mpp_dataset', type=str, default='bbbp')
    parser.add_argument('--num_shots', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=4)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    # parser = Stage2DM.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision='bf16',
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

