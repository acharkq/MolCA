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

# from lavis.common.registry import registry
# from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    # Blip2Base,
    disabled_train,
)
from model.blip2 import Blip2Base
from transformers import AutoTokenizer
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM


opt_model_list = [
    "facebook/galactica-125m",
    "facebook/galactica-1.3b",
    "facebook/galactica-6.7b",
    "facebook/galactica-6.7b",
    "facebook/galactica-30b",
]

# @registry.register_model("blip2")
# @registry.register_model("blip2_feature_extractor")
class Blip2OPT(Blip2Base):
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
        opt_model="facebook/galactica-1.3b",
        prompt="",
    ):
        super().__init__()
        self.graph_encoder, self.ln_graph = self.init_graph_encoder(gin_num_layers, gin_hidden_dim, gin_drop_ratio, use_bn)
        self.tune_gnn = tune_gnn
        if not tune_gnn:
            for name, param in self.graph_encoder.named_parameters():
                param.requires_grad = False
            self.graph_encoder = self.graph_encoder.eval()
            self.graph_encoder.train = disabled_train
            logging.info("freeze graph encoder")
        
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
        
        self.opt_model = OPTForCausalLM.from_pretrained(
            opt_model, torch_dtype=torch.float16
        )
        for name, param in self.opt_model.named_parameters():
            param.requires_grad = False
        
        ## fixme: this is different from the original BLIP2
        # self.eos_token_id = self.opt_tokenizer(
        #     "\n", add_special_tokens=False
        # ).input_ids[0]

        self.opt_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.opt_model.config.hidden_size
        )
        
        ## fixme: no prompt yet
        self.prompt = prompt
        prompt_tokens = self.opt_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, batch):
        graph, text, mask = batch
        batch_node, batch_mask = self.graph_encoder(graph)
        if not self.tune_gnn:
            batch_node = batch_node.detach()
        batch_node = self.ln_graph(batch_node, batch_mask)
        device = batch_node.device
        query_tokens = self.query_tokens.expand(batch_node.shape[0], -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=batch_node,
            encoder_attention_mask=batch_mask, # fixme: check whether this mask is correct
            return_dict=True,
        )
        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)
        targets = text.masked_fill(
            text == self.opt_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt
        
        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)
        inputs_embeds = self.opt_model.model.decoder.embed_tokens(text)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, mask], dim=1)
        
        outputs = self.opt_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            labels=targets,
        )
        loss = outputs.loss
        return {"loss": loss}

    