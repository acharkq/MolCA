"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, LoraConfig, TaskType, PeftModel
from ogb.utils import smiles2graph
from torch_geometric.loader.dataloader import Collater
from torch_geometric.data import Data
import numpy as np
# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
# from lavis.models.blip2_models.modeling_opt import OPTForCausalLM
from model.modeling_opt import OPTForSequenceClassification
# from transformers import OPTForSequenceClassification


opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]





class MolCAOPTClf(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """
    def __init__(
        self,
        bert_name,
        gin_num_layers,
        gin_hidden_dim,
        gin_drop_ratio,
        tune_gnn=False,
        num_query_token=32,
        cross_attention_freq=2,
        use_bn=False,
        llm_tune='freeze',
        peft_dir='',
        opt_model="facebook/galactica-1.3b",
        prompt="",
        args=None,
    ):
        super().__init__()
        self.args = args

        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio, use_bn)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
        self.num_query_token = num_query_token
        self.Qformer, self.query_tokens = self.init_Qformer(bert_name, num_query_token, self.graph_encoder.num_features, cross_attention_freq
        )
        ### remove the unused parameters
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        ## initialize opt model
        self.opt_tokenizer = AutoTokenizer.from_pretrained(opt_model, use_fast=False, padding_side='right')
        self.opt_tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.opt_tokenizer.add_tokens('<mol>') # molecule placeholder
        self.mol_token = '<mol>'
        self.opt_tokenizer.mol_token_id = self.opt_tokenizer("<mol>", add_special_tokens=False).input_ids[0]

        self.collater = Collater([], [])
        
        if opt_model == 'facebook/galactica-125m':
            self.opt_model = OPTForSequenceClassification.from_pretrained(opt_model, num_labels=args.num_labels)
        else:
            # self.opt_model = OPTForSequenceClassification.from_pretrained(opt_model, torch_dtype=torch.float16, num_labels=args.num_labels)
            self.opt_model = OPTForSequenceClassification.from_pretrained(opt_model, num_labels=args.num_labels)
        self.opt_model.resize_token_embeddings(len(self.opt_tokenizer)) ## this will cause bug when full fine-tuning the opt model

        self.llm_tune = llm_tune
        if llm_tune == 'lora':
            if peft_dir:
                self.opt_model = PeftModel.from_pretrained(self.opt_model, peft_dir, is_trainable=True)
            else:
                if self.args.peft_config:
                    peft_config = LoraConfig(**LoraConfig.from_json_file(self.args.peft_config))
                else:
                    peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout)
                self.peft_config = peft_config
                self.opt_model = get_peft_model(self.opt_model, peft_config)
                self.opt_model.print_trainable_parameters()
        elif llm_tune == 'freeze':
            for name, param in self.opt_model.named_parameters():
                param.requires_grad = False
        elif llm_tune == 'full':
            pass
        else:
            raise NotImplementedError()

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        self.prompt = prompt
        self.loss_func = nn.BCEWithLogitsLoss(reduction="none")

    
    def forward(self, batch, mode='train'):
        # with self.maybe_autocast():
        graphs, prompt_tokens = batch
        graph_embeds, graph_masks = self.graph_encoder(graphs)
        if not self.tune_gnn:
            graph_embeds = graph_embeds.detach()
        graph_embeds = self.ln_graph(graph_embeds, graph_masks)
        device = graph_embeds.device
        query_tokens = self.query_tokens.expand(graph_embeds.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=graph_embeds,
            encoder_attention_mask=graph_masks, # fixme: check whether this mask is correct
            return_dict=True,
        )
        mol_tokens = self.opt_proj(query_output.last_hidden_state)
    
        prompt_embeds = self.opt_model.get_input_embeddings()(prompt_tokens.input_ids)
        prompt_embeds[prompt_tokens.is_mol_token] = mol_tokens.flatten(0, 1)
        
        outputs = self.opt_model(
            inputs_embeds=prompt_embeds,
            attention_mask=prompt_tokens.attention_mask,
            return_dict=True,
        )
        logits = outputs.logits
        if mode == 'inference':
            return logits
        
        ## add together the valid loss
        loss_mat = self.loss_func(logits, ((graphs.y + 1) / 2).to(logits.dtype))
        is_valid = graphs.y.abs() > 0
        loss_mat = torch.where(is_valid, loss_mat, torch.zeros_like(loss_mat))  # shape = [N, C]
        loss = torch.sum(loss_mat)/torch.sum(is_valid)
        return {"loss": loss}
