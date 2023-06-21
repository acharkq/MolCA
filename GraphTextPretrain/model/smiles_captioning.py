import os
from typing import Any, Dict, List, Mapping, Union
import torch
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
from peft import LoraConfig, TaskType, PeftModel, get_peft_model
from transformers import BertTokenizer, AutoTokenizer, OPTForCausalLM, LlamaForCausalLM


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self
    

class SmilesCaptionLM(pl.LightningModule):
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        if self.llm_tune != 'full':
            to_be_removed = []
            for key in checkpoint['state_dict']:
                if key.startswith('llm_model'):
                    to_be_removed.append(key)
            for key in to_be_removed:
                checkpoint['state_dict'].pop(key)

        if self.llm_tune == 'lora' and (self.current_epoch + 1) % 10 == 0:
            if self.local_rank == 0: # manually fix a bug in peft module
                peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=self.args.lora_r, lora_alpha=self.args.lora_alpha, lora_dropout=self.args.lora_dropout)
                self.llm_model.peft_config['default'] = peft_config
                self.llm_model.save_pretrained(os.path.join(self.logger.save_dir, f'lora_epoch_{self.current_epoch}'))
        return super().on_save_checkpoint(checkpoint)
    
    def __init__(self, args):
        super().__init__()
        if isinstance(args, dict):
            args = AttrDict(**args)
        self.args = args
        self.caption_eval_epoch = args.caption_eval_epoch
        self.do_sample = args.do_sample
        self.num_beams = args.num_beams
        self.max_len = args.max_len
        self.min_len = args.min_len
        self.llm_tune = args.llm_tune
        self.llm_name = args.llm_name
        
        ## initialize opt model
        self.tokenizer = AutoTokenizer.from_pretrained(self.llm_name, use_fast=False, padding_side='right')
        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        
        if self.llm_name == 'facebook/galactica-125m':
            self.llm_model = OPTForCausalLM.from_pretrained(self.llm_name)
        else:
            self.llm_model = OPTForCausalLM.from_pretrained(self.llm_name, torch_dtype=torch.float16)
        self.llm_model.resize_token_embeddings(len(self.tokenizer) + 1) # for the special placeholder token

        if self.llm_tune == 'lora':
            if args.peft_dir:
                self.llm_model = PeftModel.from_pretrained(self.llm_model, args.peft_dir, is_trainable=True)
            else:
                if args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.CAUSAL_LM, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.llm_model = get_peft_model(self.llm_model, peft_config)
                self.llm_model.print_trainable_parameters()
        elif args.llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        ## fixme: this is different from the original BLIP2
        self.eos_token_id = self.tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]
        
        ## fixme: no prompt yet
        self.prompt = args.prompt
        
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
        self.bert_tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        self.save_hyperparameters(args)
    
    def configure_optimizers(self):
        self.trainer.reset_train_dataloader()
        warmup_steps = min(len(self.trainer.train_dataloader), self.args.warmup_steps)
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(optimizer, self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, warmup_steps)
        elif self.args.scheduler == 'None':
            self.scheduler = None
        else:
            raise NotImplementedError()
        return optimizer
    

    def save_predictions(self, predictions, targets):
        assert len(predictions) == len(targets)
        with open(os.path.join(self.logger.log_dir, 'predictions.txt'), 'w', encoding='utf8') as f:
            for p, t in zip(predictions, targets):
                line = {'prediction': p, 'target': t}
                f.write(json.dumps(line, ensure_ascii=True) + '\n')

    @torch.no_grad()
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        if dataloader_idx == 0:
            smiles_tokens, caption_tokens = batch
            device = smiles_tokens.input_ids.device
            batch_size = smiles_tokens.input_ids.shape[0]
            attention_mask = torch.cat((smiles_tokens.attention_mask, caption_tokens.attention_mask), dim=1)
            empty_targets = torch.ones(smiles_tokens.attention_mask.size(), dtype=torch.long).to(device).fill_(-100)
            targets = caption_tokens.input_ids.masked_fill(
                caption_tokens.input_ids == self.tokenizer.pad_token_id, -100
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            input_ids = torch.cat((smiles_tokens.input_ids, caption_tokens.input_ids), dim=1)
            outputs = self.llm_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
            loss = outputs.loss
            self.log('val_loss', float(loss), batch_size=batch_size, sync_dist=True)
            return loss
        elif dataloader_idx == 1:
            if (self.current_epoch+1) % self.caption_eval_epoch != 0:
                return 
            smiles_tokens, captions = batch
            samples = {'prompt_tokens': smiles_tokens}
            ###============== Captioning Results ===================###
            predictions = self.generate(
                samples,
                do_sample=self.do_sample,
                num_beams=self.num_beams,
                max_length=self.max_len,
                min_length=self.min_len,
            )
            return predictions, captions
        else:
            raise NotImplementedError
    
    @torch.no_grad()
    def generate(
        self, 
        samples,
        do_sample=False,
        num_beams=5,
        max_length=128,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1
        ):
        prompt_tokens = samples['prompt_tokens']
        inputs_embeds = self.llm_model.get_input_embeddings()(prompt_tokens.input_ids)
        outputs = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prompt_tokens.attention_mask,
            do_sample=do_sample,
            top_p=top_p,
            temperature=temperature,
            num_beams=num_beams,
            max_length=max_length,
            min_length=min_length,
            # pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            num_return_sequences=num_captions,
            # use_cache=False,
        )
        output_text = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        output_text = [text.strip() for text in output_text]
        return output_text
    
    def validation_epoch_end(self, outputs):
        if (self.current_epoch+1) % self.caption_eval_epoch != 0:
            return 
        caption_outputs = outputs[1]
        list_predictions, list_targets = zip(*caption_outputs)
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
                caption_evaluate(all_predictions, all_targets, self.bert_tokenizer, self.max_len * 2) 
            self.log("bleu2", bleu2, sync_dist=False)
            self.log("bleu4", bleu4, sync_dist=False)
            self.log("rouge_1", rouge_1, sync_dist=False)
            self.log("rouge_2", rouge_2, sync_dist=False)
            self.log("rouge_l", rouge_l, sync_dist=False)
            self.log("meteor_score", meteor_score, sync_dist=False)
        

    def training_step(self, batch, batch_idx):
        if self.scheduler:
            self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)
        
        smiles_tokens, caption_tokens = batch
        batch_size = smiles_tokens.input_ids.shape[0]
        device = smiles_tokens.input_ids.device

        attention_mask = torch.cat((smiles_tokens.attention_mask, caption_tokens.attention_mask), dim=1)
        empty_targets = torch.ones(smiles_tokens.attention_mask.size(), dtype=torch.long).to(device).fill_(-100)
        targets = caption_tokens.input_ids.masked_fill(
            caption_tokens.input_ids == self.tokenizer.pad_token_id, -100
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        input_ids = torch.cat((smiles_tokens.input_ids, caption_tokens.input_ids), dim=1)
        outputs = self.llm_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        self.log('train_loss', float(loss), batch_size=batch_size, sync_dist=True)
        return {"loss": loss}

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
        parser.add_argument('--llm_name', type=str, default="facebook/galactica-1.3b")
        # parser.add_argument('--prompt', type=str, default='a molecule of ')
        parser.add_argument('--num_beams', type=int, default=5)
        parser.add_argument('--do_sample', action='store_true', default=False)
        parser.add_argument('--max_len', type=int, default=256)
        parser.add_argument('--min_len', type=int, default=8)
        parser.add_argument('--llm_tune', type=str, default='freeze')
        parser.add_argument('--peft_dir', type=str, default='')

        ## quantization
        parser.add_argument('--load_in_8bit', action='store_true', default=False)

        parser.add_argument('--save_every_n_epochs', type=int, default=0)
        ## lora config
        parser.add_argument('--lora_r', type=int, default=8)
        parser.add_argument('--lora_alpha', type=int, default=32)
        parser.add_argument('--lora_dropout', type=int, default=0.1)
        parser.add_argument('--peft_config', type=str, default=None)

        # optimization
        parser.add_argument('--reaction_weight', type=float, default=1.0)
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