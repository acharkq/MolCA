'''
model for molecule property prediction
'''

import os
from typing import Any, Dict, List, Mapping, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
import torch
from model.blip2_opt import Blip2OPT
from model.blip2_llama import Blip2Llama
import pytorch_lightning as pl
import torch.nn as nn
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler
import json
from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from tqdm import tqdm
import numpy as np
import torch.distributed as dist
from peft import LoraConfig, TaskType

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def load_ignore_unexpected(model, state_dict):
    keys = set(model.state_dict().keys())
    # print(state_dict.keys())
    state_dict = {k: v for k, v in state_dict.items() if k in keys}
    model.load_state_dict(state_dict, strict=True)
    

def get_module_state_dict(state_dict, module_name):
    module_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith(module_name):
            key = key[len(module_name) + 1:]
            if key == '':
                return value
            module_state_dict[key] = value
    return module_state_dict

class Blip2MPP(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        to_be_removed = []
        for key in checkpoint['state_dict']:
            if key.startswith('blip2opt.opt_model') or key.startswith('blip2opt.llm_model'):
                to_be_removed.append(key)
        for key in to_be_removed:
            checkpoint['state_dict'].pop(key)
        if self.lora_tuning and (self.current_epoch + 1) % 10 == 0:
            if self.local_rank == 0: # manually fix a bug in peft module
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1)
                if hasattr(self.blip2opt, 'opt_model'):
                    self.blip2opt.opt_model.peft_config['default'] = peft_config
                    self.blip2opt.opt_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
                elif hasattr(self.blip2opt, 'llm_model'):
                    self.blip2opt.llm_model.peft_config['default'] = peft_config
                    self.blip2opt.llm_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)

        self.args = args
        self.nucleus_sampling = args.nucleus_sampling
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.lora_tuning = args.lora_tuning
        if args.opt_model.find('galactica') >= 0:
            self.blip2opt = Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.use_bn, args.lora_tuning, args.peft_dir, args.opt_model, args.prompt, args)
        elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
            self.blip2opt = Blip2Llama(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.use_bn, args.lora_tuning, args.peft_dir, args.opt_model, args.prompt, args)
        else:
            raise NotImplementedError()
        self.tokenizer = self.blip2opt.init_tokenizer()
        self.save_hyperparameters(args)

    def load_from_stage1_checkpoint(self, path):
        ckpt = torch.load(path, map_location='cpu')
        state_dict = ckpt['state_dict']
        qformer_dict = get_module_state_dict(state_dict, 'blip2qformer.Qformer')
        ln_graph_dict = get_module_state_dict(state_dict, 'blip2qformer.ln_graph')
        qs_weight = get_module_state_dict(state_dict, 'blip2qformer.query_tokens')
        load_ignore_unexpected(self.blip2opt.Qformer, qformer_dict)
        self.blip2opt.ln_graph.load_state_dict(ln_graph_dict)
        self.blip2opt.query_tokens.data.copy_(qs_weight)
        return self
    
    def load_from_blip(self, path):
        ckpt = torch.load(path, map_location='cpu')
        self.blip2opt.load_state_dict(ckpt['state_dict'], strict=False)
        return self
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        # warmup_steps = min(self.args.warmup_steps, len(self.train_dataloader))
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, self.args.warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer

    def test_epoch_end(self, outputs):
        list_predictions, list_targets = zip(*outputs)
        predictions = [i for ii in list_predictions for i in ii]
        targets = [i for ii in list_targets for i in ii]

        all_predictions = [None for _ in range(self.trainer.world_size)]
        all_targets = [None for _ in range(self.trainer.world_size)]

        dist.all_gather_object(all_predictions, predictions)
        dist.all_gather_object(all_targets, targets)
        if self.global_rank == 0:
            all_predictions = [i for ii in all_predictions for i in ii]
            all_targets = [i for ii in all_targets for i in ii]
            self.save_predictions(all_predictions, all_targets)
            ## fixme: I am not sure if the max length is the same as previous experiments
            bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = \
                caption_evaluate(all_predictions, all_targets, self.tokenizer, self.max_len * 2) 
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        graphs, prompt_tokens, texts = batch
        ###============== Captioning Results ===================###
        samples = {'graphs': graphs, 'prompt_tokens': prompt_tokens}
        predictions = self.blip2opt.generate(
            samples, 
            use_nucleus_sampling=self.nucleus_sampling,
            num_beams=self.num_beams,
            max_length=self.max_len * 2,
            min_length=self.min_len
        )
        return predictions, texts

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            _, _, prompt_lens = batch
            batch_size = prompt_lens.shape[0]
            loss = self.blip2opt(batch)
            ###============== Overall Loss ===================###
            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            return loss['loss']
        elif dataloader_idx == 1:
            reaction_tokens, _, _ = batch
            batch_size = reaction_tokens.input_ids.shape[0]
            loss = self.blip2opt.forward_reaction(batch)
            ###============== Overall Loss ===================###
            self.log("reaction loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            return loss['loss']
        else:
            raise NotImplementedError
        
    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        if isinstance(batch, list) and len(batch) == 2:
            molecule_batch, reaction_batch = batch
            batch_size = molecule_batch[-1].size(0)
            ###============== molecule Loss ===================###
            molecule_loss = self.blip2opt(molecule_batch)['loss']
            self.log("molecule loss", float(molecule_loss), batch_size=batch_size, sync_dist=True)
            
            ###============== reaction Loss ===================###
            reaction_loss = self.blip2opt.forward_reaction(reaction_batch)['loss']
            self.log("reaction loss", float(reaction_loss), batch_size=batch_size, sync_dist=True)

            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return molecule_loss + reaction_loss
        else:
            batch_size = batch[-1].size(0)
            ###============== Overall Loss ===================###
            loss = self.blip2opt(batch)
            self.log("molecule loss", float(loss['loss']), batch_size=batch_size, sync_dist=True)
            self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size, sync_dist=True)
            return loss['loss']

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--tune_gnn', action='store_true', default=False)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=8)
        # OPT
        parser.add_argument('--opt_model', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--nucleus_sampling', action='store_true', default=True)
        parser.add_argument('--max_len', type=int, default=128)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--lora_tuning', action='store_true', default=False)
        parser.add_argument('--peft_dir', type=str, default='')

        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-6, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=1000, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_cosine_lr', help='type of scheduler') # or linear_warmup_step_lr
        parser.add_argument('--stage1_path', type=str, default='')
        parser.add_argument('--stage2_path', type=str, default='')
        parser.add_argument('--init_checkpoint', type=str, default='')
        parser.add_argument('--caption_eval_epoch', type=int, default=10)
        return parent_parser


def caption_evaluate(predictions, targets, tokenizer, text_trunc_length):
    meteor_scores = []
    references = []
    hypotheses = []
    for gt, out in tqdm(zip(targets, predictions)):
        gt_tokens = tokenizer.tokenize(gt, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        gt_tokens = list(filter(('[PAD]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[CLS]').__ne__, gt_tokens))
        gt_tokens = list(filter(('[SEP]').__ne__, gt_tokens))

        out_tokens = tokenizer.tokenize(out, truncation=True, max_length=text_trunc_length,
                                            padding='max_length')
        out_tokens = list(filter(('[PAD]').__ne__, out_tokens))
        out_tokens = list(filter(('[CLS]').__ne__, out_tokens))
        out_tokens = list(filter(('[SEP]').__ne__, out_tokens))

        references.append([gt_tokens])
        hypotheses.append(out_tokens)

        mscore = meteor_score([gt_tokens], out_tokens)
        meteor_scores.append(mscore)

    bleu2 = corpus_bleu(references, hypotheses, weights=(.5,.5))
    bleu4 = corpus_bleu(references, hypotheses, weights=(.25,.25,.25,.25))
    bleu2 *= 100
    bleu4 *= 100

    print('BLEU-2 score:', bleu2)
    print('BLEU-4 score:', bleu4)
    _meteor_score = np.mean(meteor_scores)
    _meteor_score *= 100
    print('Average Meteor score:', _meteor_score)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    rouge_scores = []

    references = []
    hypotheses = []

    for gt, out in tqdm(zip(targets, predictions)):
        rs = scorer.score(out, gt)
        rouge_scores.append(rs)

    print('ROUGE score:')
    rouge_1 = np.mean([rs['rouge1'].fmeasure for rs in rouge_scores]) * 100
    rouge_2 = np.mean([rs['rouge2'].fmeasure for rs in rouge_scores]) * 100
    rouge_l = np.mean([rs['rougeL'].fmeasure for rs in rouge_scores]) * 100
    print('rouge1:', rouge_1)
    print('rouge2:', rouge_2)
    print('rougeL:', rouge_l)
    return bleu2, bleu4, rouge_1, rouge_2, rouge_l, _meteor_score