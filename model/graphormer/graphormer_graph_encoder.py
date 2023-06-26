# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
from fairseq.modules import FairseqDropout, LayerDropModuleList, LayerNorm
from fairseq.modules.quant_noise import quant_noise as apply_quant_noise_

from .multihead_attention import MultiheadAttention
from .graphormer_layers import GraphNodeFeature, GraphAttnBias
from .graphormer_graph_encoder_layer import GraphormerGraphEncoderLayer
# from data_provider.pred_datamodule import get_dataset
import pytorch_lightning as pl


def init_graphormer_params(module):
    """
    Initialize the weights specific to the Graphormer Model.
    """

    def normal_(data):
        # with FSDP, module params will be on CUDA, so we cast them back to CPU
        # so that the RNG is consistent with and without FSDP
        data.copy_(data.cpu().normal_(mean=0.0, std=0.02).to(data.device))

    if isinstance(module, nn.Linear):
        normal_(module.weight.data)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        normal_(module.weight.data)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        normal_(module.q_proj.weight.data)
        normal_(module.k_proj.weight.data)
        normal_(module.v_proj.weight.data)


class GraphormerGraphEncoder(nn.Module):
    def __init__(
        self,
        num_atoms: int,
        num_in_degree: int,
        num_out_degree: int,
        num_edges: int,
        num_spatial: int,
        num_edge_dis: int,
        edge_type: str,
        multi_hop_max_dist: int,
        num_encoder_layers: int = 12,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 768,
        num_attention_heads: int = 32,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        layerdrop: float = 0.0,
        encoder_normalize_before: bool = False,
        pre_layernorm: bool = False,
        apply_graphormer_init: bool = False,
        activation_fn: str = "gelu",
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
        traceable: bool = False,
        q_noise: float = 0.0,
        qn_block_size: int = 8,
    ) -> None:

        super().__init__()
        self.dropout_module = FairseqDropout(
            dropout, module_name=self.__class__.__name__
        )
        self.layerdrop = layerdrop
        self.embedding_dim = embedding_dim
        self.apply_graphormer_init = apply_graphormer_init
        self.traceable = traceable

        self.graph_node_feature = GraphNodeFeature(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_in_degree=num_in_degree,
            num_out_degree=num_out_degree,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.graph_attn_bias = GraphAttnBias(
            num_heads=num_attention_heads,
            num_atoms=num_atoms,
            num_edges=num_edges,
            num_spatial=num_spatial,
            num_edge_dis=num_edge_dis,
            edge_type=edge_type,
            multi_hop_max_dist=multi_hop_max_dist,
            hidden_dim=embedding_dim,
            n_layers=num_encoder_layers,
        )

        self.embed_scale = embed_scale

        if q_noise > 0:
            self.quant_noise = apply_quant_noise_(
                nn.Linear(self.embedding_dim, self.embedding_dim, bias=False),
                q_noise,
                qn_block_size,
            )
        else:
            self.quant_noise = None

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        if pre_layernorm:
            self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

        if self.layerdrop > 0.0:
            self.layers = LayerDropModuleList(p=self.layerdrop)
        else:
            self.layers = nn.ModuleList([])
        self.layers.extend(
            [
                self.build_graphormer_graph_encoder_layer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout_module.p,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    export=export,
                    q_noise=q_noise,
                    qn_block_size=qn_block_size,
                    pre_layernorm=pre_layernorm,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        # Apply initialization of model params after building the model
        if self.apply_graphormer_init:
            self.apply(init_graphormer_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            raise NotImplementedError("Freezing embeddings is not implemented yet.")

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def build_graphormer_graph_encoder_layer(
        self,
        embedding_dim,
        ffn_embedding_dim,
        num_attention_heads,
        dropout,
        attention_dropout,
        activation_dropout,
        activation_fn,
        export,
        q_noise,
        qn_block_size,
        pre_layernorm,
    ):
        return GraphormerGraphEncoderLayer(
            embedding_dim=embedding_dim,
            ffn_embedding_dim=ffn_embedding_dim,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
            activation_fn=activation_fn,
            export=export,
            q_noise=q_noise,
            qn_block_size=qn_block_size,
            pre_layernorm=pre_layernorm,
        )

    def forward(
        self,
        batched_data,
        perturb=None,
        last_state_only: bool = False,
        token_embeddings: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        is_tpu = False
        # compute padding mask. This is needed for multi-head attention
        data_x = batched_data["x"]
        n_graph, n_node = data_x.size()[:2]  # size的前两维
        padding_mask = (data_x[:, :, 0]).eq(0)  # B x T x 1  对两个Tensor进行逐元素的比较，若相同位置的两个元素相同，则返回True
        padding_mask_cls = torch.zeros(n_graph, 1, device=padding_mask.device, dtype=padding_mask.dtype)  # [cls]这个特殊token的mask
        padding_mask = torch.cat((padding_mask_cls, padding_mask), dim=1)
        # B x (T+1) x 1

        if token_embeddings is not None:
            x = token_embeddings
        else:
            x = self.graph_node_feature(batched_data)

        if perturb is not None:
            #ic(torch.mean(torch.abs(x[:, 1, :])))
            #ic(torch.mean(torch.abs(perturb)))
            x[:, 1:, :] += perturb

        # x: B x T x C

        attn_bias = self.graph_attn_bias(batched_data)

        if self.embed_scale is not None:
            x = x * self.embed_scale

        if self.quant_noise is not None:
            x = self.quant_noise(x)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        x = self.dropout_module(x)

        # account for padding while computing the representation

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        for layer in self.layers:
            x, _ = layer(
                x,
                self_attn_padding_mask=padding_mask,
                self_attn_mask=attn_mask,
                self_attn_bias=attn_bias,
            )
            if not last_state_only:
                inner_states.append(x)

        graph_rep = x[0, :, :]  # 即[cls] token的输出，代表图的表征向量

        if last_state_only:
            inner_states = [x]

        if self.traceable:
            return torch.stack(inner_states), graph_rep
        else:
            return inner_states, graph_rep


# class GraphormerGraphPred(nn.Module):
#     def __init__(self, configs):
#         super(GraphormerGraphPred, self).__init__()
#         self.gnn = GraphormerGraphEncoder(
#             configs.num_atoms,  # int
#             configs.num_in_degree,  # int
#             configs.num_out_degree,  # int
#             configs.num_edges,  # int
#             configs.num_spatial,  # int
#             configs.num_edge_dis,  # int
#             configs.edge_type,  # str
#             configs.multi_hop_max_dist,
#             configs.num_encoder_layers,
#             configs.embedding_dim,
#             configs.ffn_embedding_dim,
#             configs.num_attention_heads,
#             configs.dropout,
#             configs.attention_dropout,
#             configs.activation_dropout,
#             configs.layerdrop,
#             configs.encoder_normalize_before,
#             configs.pre_layernorm,
#             configs.apply_graphormer_init,
#             configs.activation_fn,
#             configs.embed_scale,
#             configs.freeze_embeddings,
#             configs.n_trans_layers_to_freeze,
#             configs.export,
#             configs.traceable,
#             configs.q_noise,
#             configs.qn_block_size,
#         )

#         self.out_proj = nn.Linear(configs.embedding_dim, get_dataset(configs.dataset_name)['num_class'])

#     def forward(self, graph, perturb=None, last_state_only=False, token_embeddings=None, attn_mask=None):
#         _, y = self.gnn(graph, perturb, last_state_only, token_embeddings, attn_mask)
#         y = self.out_proj(y)
#         return y


# class Graphormer(pl.LightningModule):
#     def __init__(
#         self,
#         n_layers,
#         num_heads,
#         hidden_dim,
#         dropout_rate,
#         intput_dropout_rate,
#         weight_decay,
#         ffn_dim,
#         dataset_name,
#         warmup_updates,
#         tot_updates,
#         peak_lr,
#         end_lr,
#         edge_type,
#         multi_hop_max_dist,
#         attention_dropout_rate,
#         flag=False,
#         flag_m=3,
#         flag_step_size=1e-3,
#         flag_mag=1e-3,
#     ):
#         super().__init__()
#         self.save_hyperparameters()

#         self.num_heads = num_heads
#         if dataset_name == 'ZINC':
#             self.atom_encoder = nn.Embedding(64, hidden_dim, padding_idx=0)
#             self.edge_encoder = nn.Embedding(64, num_heads, padding_idx=0)
#             self.edge_type = edge_type
#             if self.edge_type == 'multi_hop':
#                 self.edge_dis_encoder = nn.Embedding(
#                     40 * num_heads * num_heads, 1)
#             self.spatial_pos_encoder = nn.Embedding(40, num_heads, padding_idx=0)
#             self.in_degree_encoder = nn.Embedding(
#                 64, hidden_dim, padding_idx=0)
#             self.out_degree_encoder = nn.Embedding(
#                 64, hidden_dim, padding_idx=0)
#         else:
#             self.atom_encoder = nn.Embedding(
#                 512 * 9 + 1, hidden_dim, padding_idx=0)
#             self.edge_encoder = nn.Embedding(
#                 512 * 3 + 1, num_heads, padding_idx=0)
#             self.edge_type = edge_type
#             if self.edge_type == 'multi_hop':
#                 self.edge_dis_encoder = nn.Embedding(
#                     128 * num_heads * num_heads, 1)
#             self.spatial_pos_encoder = nn.Embedding(512, num_heads, padding_idx=0)
#             self.in_degree_encoder = nn.Embedding(
#                 512, hidden_dim, padding_idx=0)
#             self.out_degree_encoder = nn.Embedding(
#                 512, hidden_dim, padding_idx=0)

#         self.input_dropout = nn.Dropout(intput_dropout_rate)
#         encoders = [EncoderLayer(hidden_dim, ffn_dim, dropout_rate, attention_dropout_rate, num_heads)
#                     for _ in range(n_layers)]
#         self.layers = nn.ModuleList(encoders)
#         self.final_ln = nn.LayerNorm(hidden_dim)

#         if dataset_name == 'PCQM4M-LSC':
#             self.out_proj = nn.Linear(hidden_dim, 1)
#         else:
#             self.downstream_out_proj = nn.Linear(
#                 hidden_dim, get_dataset(dataset_name)['num_class'])

#         self.graph_token = nn.Embedding(1, hidden_dim)
#         self.graph_token_virtual_distance = nn.Embedding(1, num_heads)

#         self.evaluator = get_dataset(dataset_name)['evaluator']
#         self.metric = get_dataset(dataset_name)['metric']
#         self.loss_fn = get_dataset(dataset_name)['loss_fn']
#         self.dataset_name = dataset_name

#         self.warmup_updates = warmup_updates
#         self.tot_updates = tot_updates
#         self.peak_lr = peak_lr
#         self.end_lr = end_lr
#         self.weight_decay = weight_decay
#         self.multi_hop_max_dist = multi_hop_max_dist

#         self.flag = flag
#         self.flag_m = flag_m
#         self.flag_step_size = flag_step_size
#         self.flag_mag = flag_mag
#         self.hidden_dim = hidden_dim
#         self.automatic_optimization = not self.flag
#         self.apply(lambda module: init_params(module, n_layers=n_layers))

#     def forward(self, batched_data, perturb=None):
#         attn_bias, spatial_pos, x = batched_data.attn_bias, batched_data.spatial_pos, batched_data.x
#         in_degree, out_degree = batched_data.in_degree, batched_data.in_degree
#         edge_input, attn_edge_type = batched_data.edge_input, batched_data.attn_edge_type
#         # graph_attn_bias
#         n_graph, n_node = x.size()[:2]
#         graph_attn_bias = attn_bias.clone()
#         graph_attn_bias = graph_attn_bias.unsqueeze(1).repeat(
#             1, self.num_heads, 1, 1)  # [n_graph, n_head, n_node+1, n_node+1]

#         # spatial pos
#         # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
#         spatial_pos_bias = self.spatial_pos_encoder(spatial_pos).permute(0, 3, 1, 2)
#         graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
#                                                         :, 1:, 1:] + spatial_pos_bias
#         # reset spatial pos here
#         t = self.graph_token_virtual_distance.weight.view(1, self.num_heads, 1)
#         graph_attn_bias[:, :, 1:, 0] = graph_attn_bias[:, :, 1:, 0] + t
#         graph_attn_bias[:, :, 0, :] = graph_attn_bias[:, :, 0, :] + t

#         # edge feature
#         if self.edge_type == 'multi_hop':
#             spatial_pos_ = spatial_pos.clone()
#             spatial_pos_[spatial_pos_ == 0] = 1  # set pad to 1
#             # set 1 to 1, x > 1 to x - 1
#             spatial_pos_ = torch.where(spatial_pos_ > 1, spatial_pos_ - 1, spatial_pos_)
#             if self.multi_hop_max_dist > 0:
#                 spatial_pos_ = spatial_pos_.clamp(0, self.multi_hop_max_dist)
#                 edge_input = edge_input[:, :, :, :self.multi_hop_max_dist, :]
#             # [n_graph, n_node, n_node, max_dist, n_head]
#             edge_input = self.edge_encoder(edge_input).mean(-2)
#             max_dist = edge_input.size(-2)
#             edge_input_flat = edge_input.permute(
#                 3, 0, 1, 2, 4).reshape(max_dist, -1, self.num_heads)
#             edge_input_flat = torch.bmm(edge_input_flat, self.edge_dis_encoder.weight.reshape(
#                 -1, self.num_heads, self.num_heads)[:max_dist, :, :])
#             edge_input = edge_input_flat.reshape(
#                 max_dist, n_graph, n_node, n_node, self.num_heads).permute(1, 2, 3, 0, 4)
#             edge_input = (edge_input.sum(-2) /
#                           (spatial_pos_.float().unsqueeze(-1))).permute(0, 3, 1, 2)
#         else:
#             # [n_graph, n_node, n_node, n_head] -> [n_graph, n_head, n_node, n_node]
#             edge_input = self.edge_encoder(
#                 attn_edge_type).mean(-2).permute(0, 3, 1, 2)

#         graph_attn_bias[:, :, 1:, 1:] = graph_attn_bias[:,
#                                                         :, 1:, 1:] + edge_input
#         graph_attn_bias = graph_attn_bias + attn_bias.unsqueeze(1)  # reset

#         # node feauture + graph token
#         node_feature = self.atom_encoder(x).sum(
#             dim=-2)           # [n_graph, n_node, n_hidden]
#         if self.flag and perturb is not None:
#             node_feature += perturb

#         node_feature = node_feature + \
#             self.in_degree_encoder(in_degree) + \
#             self.out_degree_encoder(out_degree)
#         graph_token_feature = self.graph_token.weight.unsqueeze(
#             0).repeat(n_graph, 1, 1)
#         graph_node_feature = torch.cat(
#             [graph_token_feature, node_feature], dim=1)

#         # transfomrer encoder
#         output = self.input_dropout(graph_node_feature)
#         for enc_layer in self.layers:
#             output = enc_layer(output, graph_attn_bias)
#         output = self.final_ln(output)

#         # output part
#         if self.dataset_name == 'PCQM4M-LSC':
#             # get whole graph rep
#             output = self.out_proj(output[:, 0, :])
#         else:
#             output = self.downstream_out_proj(output[:, 0, :])
#         return output

#     def training_step(self, batched_data, batch_idx):
#         if self.dataset_name == 'ogbg-molpcba':
#             if not self.flag:
#                 y_hat = self(batched_data).view(-1)
#                 y_gt = batched_data.y.view(-1).float()
#                 mask = ~torch.isnan(y_gt)
#                 loss = self.loss_fn(y_hat[mask], y_gt[mask])
#             else:
#                 y_gt = batched_data.y.view(-1).float()
#                 mask = ~torch.isnan(y_gt)

#                 def forward(perturb): return self(batched_data, perturb)
#                 model_forward = (self, forward)
#                 n_graph, n_node = batched_data.x.size()[:2]
#                 perturb_shape = (n_graph, n_node, self.hidden_dim)

#                 optimizer = self.optimizers()
#                 optimizer.zero_grad()
#                 loss, _ = flag_bounded(model_forward, perturb_shape, y_gt[mask], optimizer, batched_data.x.device, self.loss_fn,
#                                        m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag, mask=mask)
#                 self.lr_schedulers().step()

#         elif self.dataset_name == 'ogbg-molhiv':
#             if not self.flag:
#                 y_hat = self(batched_data).view(-1)
#                 y_gt = batched_data.y.view(-1).float()
#                 loss = self.loss_fn(y_hat, y_gt)
#             else:
#                 y_gt = batched_data.y.view(-1).float()
#                 def forward(perturb): return self(batched_data, perturb)
#                 model_forward = (self, forward)
#                 n_graph, n_node = batched_data.x.size()[:2]
#                 perturb_shape = (n_graph, n_node, self.hidden_dim)

#                 optimizer = self.optimizers()
#                 optimizer.zero_grad()
#                 loss, _ = flag_bounded(model_forward, perturb_shape, y_gt, optimizer, batched_data.x.device, self.loss_fn,
#                                        m=self.flag_m, step_size=self.flag_step_size, mag=self.flag_mag)
#                 self.lr_schedulers().step()
#         else:
#             y_hat = self(batched_data).view(-1)
#             y_gt = batched_data.y.view(-1)
#             loss = self.loss_fn(y_hat, y_gt)
#         self.log('train_loss', loss, sync_dist=True)
#         return loss

#     def validation_step(self, batched_data, batch_idx):
#         if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
#             y_pred = self(batched_data).view(-1)
#             y_true = batched_data.y.view(-1)
#         else:
#             y_pred = self(batched_data)
#             y_true = batched_data.y
#         return {
#             'y_pred': y_pred,
#             'y_true': y_true,
#         }

#     def validation_epoch_end(self, outputs):
#         y_pred = torch.cat([i['y_pred'] for i in outputs])
#         y_true = torch.cat([i['y_true'] for i in outputs])
#         if self.dataset_name == 'ogbg-molpcba':
#             mask = ~torch.isnan(y_true)
#             loss = self.loss_fn(y_pred[mask], y_true[mask])
#             self.log('valid_ap', loss, sync_dist=True)
#         else:
#             input_dict = {"y_true": y_true, "y_pred": y_pred}
#             try:
#                 self.log('valid_' + self.metric, self.evaluator.eval(input_dict)
#                          [self.metric], sync_dist=True)
#             except:
#                 pass

#     def test_step(self, batched_data, batch_idx):
#         if self.dataset_name in ['PCQM4M-LSC', 'ZINC']:
#             y_pred = self(batched_data).view(-1)
#             y_true = batched_data.y.view(-1)
#         else:
#             y_pred = self(batched_data)
#             y_true = batched_data.y
#         return {
#             'y_pred': y_pred,
#             'y_true': y_true,
#             'idx': batched_data.idx,
#         }

#     def test_epoch_end(self, outputs):
#         y_pred = torch.cat([i['y_pred'] for i in outputs])
#         y_true = torch.cat([i['y_true'] for i in outputs])
#         if self.dataset_name == 'PCQM4M-LSC':
#             result = y_pred.cpu().float().numpy()
#             idx = torch.cat([i['idx'] for i in outputs])
#             torch.save(result, 'y_pred.pt')
#             torch.save(idx, 'idx.pt')
#             exit(0)
#         input_dict = {"y_true": y_true, "y_pred": y_pred}
#         self.log('test_' + self.metric, self.evaluator.eval(input_dict)
#                  [self.metric], sync_dist=True)

#     def configure_optimizers(self):
#         optimizer = torch.optim.AdamW(
#             self.parameters(), lr=self.peak_lr, weight_decay=self.weight_decay)
#         lr_scheduler = {
#             'scheduler': PolynomialDecayLR(
#                 optimizer,
#                 warmup_updates=self.warmup_updates,
#                 tot_updates=self.tot_updates,
#                 lr=self.peak_lr,
#                 end_lr=self.end_lr,
#                 power=1.0,
#             ),
#             'name': 'learning_rate',
#             'interval': 'step',
#             'frequency': 1,
#         }
#         return [optimizer], [lr_scheduler]

#     @staticmethod
#     def add_model_specific_args(parent_parser):
#         parser = parent_parser.add_argument_group("Graphormer")
#         parser.add_argument('--n_layers', type=int, default=12)
#         parser.add_argument('--num_heads', type=int, default=32)
#         parser.add_argument('--hidden_dim', type=int, default=512)
#         parser.add_argument('--ffn_dim', type=int, default=512)
#         parser.add_argument('--intput_dropout_rate', type=float, default=0.1)
#         parser.add_argument('--dropout_rate', type=float, default=0.1)
#         parser.add_argument('--weight_decay', type=float, default=0.01)
#         parser.add_argument('--attention_dropout_rate',
#                             type=float, default=0.1)
#         parser.add_argument('--checkpoint_path', type=str, default='')
#         parser.add_argument('--warmup_updates', type=int, default=60000)
#         parser.add_argument('--tot_updates', type=int, default=1000000)
#         parser.add_argument('--peak_lr', type=float, default=2e-4)
#         parser.add_argument('--end_lr', type=float, default=1e-9)
#         parser.add_argument('--edge_type', type=str, default='multi_hop')
#         parser.add_argument('--validate', action='store_true', default=False)
#         parser.add_argument('--test', action='store_true', default=False)
#         parser.add_argument('--flag', action='store_true')
#         parser.add_argument('--flag_m', type=int, default=3)
#         parser.add_argument('--flag_step_size', type=float, default=1e-3)
#         parser.add_argument('--flag_mag', type=float, default=1e-3)
#         return parent_parser