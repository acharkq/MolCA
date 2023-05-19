import torch
import torch.nn as nn
from model.gin_model import GNN
from model.bert import MMEncoder
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
from torch.optim import lr_scheduler


def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
    """Warmup the learning rate"""
    lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
    """Decay the learning rate"""
    lr = max(min_lr, init_lr * (decay_rate**epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class GINSimclr(pl.LightningModule):
    def __init__(
            self,
            temperature,
            gin_hidden_dim,
            gin_num_layers,
            drop_ratio,
            graph_pooling,
            graph_self,
            declip,
            gtm,
            lm,
            bert_hidden_dim,
            pretrain,
            projection_dim,
            weight_decay,
            init_lr,
            min_lr,
            warmup_lr,
            warmup_steps,
            lr_decay_rate,
    ):
        super().__init__()
        self.declip = declip
        self.gtm = gtm
        self.lm = lm
        self.init_lr = init_lr
        self.min_lr = min_lr
        self.warmup_lr = warmup_lr
        self.warmup_steps = warmup_steps
        self.lr_decay_rate = lr_decay_rate

        self.save_hyperparameters()
        self.temperature = temperature
        self.gin_hidden_dim = gin_hidden_dim
        self.gin_num_layers = gin_num_layers
        self.drop_ratio = drop_ratio
        self.graph_pooling = graph_pooling
        self.graph_self = graph_self

        self.bert_hidden_dim = bert_hidden_dim
        self.pretrain = pretrain

        self.projection_dim = projection_dim
        self.weight_decay = weight_decay

        self.graph_encoder = GNN(
            num_layer=self.gin_num_layers,
            emb_dim=self.gin_hidden_dim,
            gnn_type='gin',
            drop_ratio=self.drop_ratio,
            JK='last',
        )
        self.graph_encoder.cat_grep = False
        ckpt = torch.load('gin_pretrained/graphcl_80.pth', map_location=torch.device('cpu'))
        missing_keys, unexpected_keys = self.graph_encoder.load_state_dict(ckpt, strict=False)
        if len(missing_keys) or len(unexpected_keys):
            print(missing_keys)
            print(unexpected_keys)

        self.text_encoder = MMEncoder(pretrain=self.pretrain, gtm=gtm, lm=lm, graph_dim=gin_hidden_dim)

        self.graph_proj_head = nn.Sequential(
          nn.Linear(self.gin_hidden_dim, self.gin_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.gin_hidden_dim, self.projection_dim)
        )

        self.text_proj_head = nn.Sequential(
          nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
          nn.ReLU(inplace=True),
          nn.Linear(self.bert_hidden_dim, self.projection_dim)
        )

        if gtm:
            self.gtm_head = nn.Sequential(
                nn.Linear(self.bert_hidden_dim, self.bert_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.bert_hidden_dim, 2)
            )

    def forward(self):
        print("forward not defined")
        exit(0)

    def contrast(self, features_graph, features_text):
        batch_size = features_graph.size(0)

        # normalized features
        features_graph = F.normalize(features_graph, dim=-1)
        features_text = F.normalize(features_text, dim=-1)

        # cosine similarity as logits
        logits_per_graph = features_graph @ features_text.t() / self.temperature
        logits_per_text = logits_per_graph.t()

        labels = torch.arange(batch_size, dtype=torch.long, device=self.device)  # 大小为B
        loss_graph = F.cross_entropy(logits_per_graph, labels)
        loss_text = F.cross_entropy(logits_per_text, labels)
        loss = (loss_graph + loss_text) / 2

        return logits_per_graph, logits_per_text, loss

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay)
        # scheduler = lr_scheduler.SequentialLR(optimizer,
        #                                       schedulers=[lr_scheduler.LinearLR(optimizer, start_factor=self.init_lr, end_factor=0.1, total_iters=80), lr_scheduler.ExponentialLR(optimizer, gamma=0.9)],
        #                                       milestones=[50])
        # return optimizer, scheduler
        return optimizer

    def on_train_epoch_start(self) -> None:
        step_lr_schedule(self.trainer.optimizers[0], self.trainer.current_epoch, self.init_lr, self.min_lr, self.lr_decay_rate)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        loss = 0.0
        batch_size = batch[-1].size(0)
        if self.declip:
            graph, graph2, text, mask, text2, mask2 = batch

            batch_node_rep, batch_mask, batch_graph_rep = self.graph_encoder(graph)
            graph_rep = self.graph_proj_head(batch_graph_rep)

            batch_node_rep2, batch_mask2, batch_graph2_rep = self.graph_encoder(graph2)
            graph2_rep = self.graph_proj_head(batch_graph2_rep)

            text_rep = self.text_encoder(text, mask)
            text_rep = self.text_proj_head(text_rep)

            text2_rep = self.text_encoder(text2, mask2)
            text2_rep = self.text_proj_head(text2_rep)

            g2t_sim, t2g_sim, loss11 = self.contrast(graph_rep, text_rep)
            _, _, loss12 = self.contrast(graph_rep, text2_rep)
            _, _, loss21 = self.contrast(graph2_rep, text_rep)
            _, _, loss22 = self.contrast(graph2_rep, text2_rep)

            if self.graph_self:
                _, _, loss_graph_self = self.contrast(graph_rep, graph2_rep)
                loss_cl = (loss11 + loss12 + loss21 + loss22 + loss_graph_self) / 5.0
            else:
                loss_cl = (loss11 + loss12 + loss21 + loss22) / 4.0
        else:
            graph, text, mask = batch

            batch_node_rep, batch_mask, batch_graph_rep = self.graph_encoder(graph)
            graph_rep = self.graph_proj_head(batch_graph_rep)

            text_rep = self.text_encoder(text, mask)
            text_rep = self.text_proj_head(text_rep)

            g2t_sim, t2g_sim, loss_cl = self.contrast(graph_rep, text_rep)

        g_emb = torch.cat([batch_graph_rep.unsqueeze(1), batch_node_rep], dim=1)
        g_mask = torch.cat([torch.ones(batch_mask.size()[:-1], dtype=torch.long).unsqueeze(1).to(self.device), batch_mask], dim=1)
        # g_emb, g_mask = g_emb[:, :64, :], g_mask[:, :64]
        self.log("val_cl_loss", loss_cl, batch_size=batch_size)
        loss += loss_cl
        ###============== Image-text Matching ===================###
        if self.gtm:
            # bs = len(graph)
            encoder_input_ids = text.clone()
            encoder_input_ids[:, 0] = self.text_encoder.tokenizer.enc_token_id

            # forward the positve image-text pair
            output_pos = self.text_encoder(encoder_input_ids, mask, g_emb, g_mask)

            # select hard negatives based on sim matrix
            with torch.no_grad():
                weights_g2t = F.softmax(g2t_sim, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)
                weights_t2g = F.softmax(t2g_sim, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb[neg_idx])
                graph_mask_neg.append(g_mask[neg_idx])
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each graph
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([mask, text_atts_neg], dim=0)

            graph_embeds_all = torch.cat([graph_embeds_neg, g_emb], dim=0)
            graph_atts_all = torch.cat([graph_mask_neg, g_mask], dim=0)

            output_neg = self.text_encoder(text_ids_all,
                                           attention_mask=text_atts_all,
                                           g_emb=graph_embeds_all,
                                           g_mask=graph_atts_all,
                                           )
            gl_embeddings = torch.cat([output_pos, output_neg], dim=0)
            gl_output = self.gtm_head(gl_embeddings)
            gtm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0).to(self.device)
            loss_gtm = F.cross_entropy(gl_output, gtm_labels)
            self.log("val_gtm_loss", loss_gtm, batch_size=batch_size)
            loss += loss_gtm

        ##================= LM ========================##
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.text_encoder.tokenizer.bos_token_id
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.text_encoder.tokenizer.pad_token_id, -100)

            decode_loss = self.text_encoder.decode_loss(decoder_input_ids,
                                                        attention_mask=mask,
                                                        g_emb=g_emb,
                                                        g_mask=g_mask,
                                                        labels=decoder_targets,
                                                        return_dict=True,
                                                        )

            loss_lm = decode_loss
            self.log("val_lm_loss", loss_lm, batch_size=batch_size)
            loss += loss_lm
        ###============== Overall Val Loss ===================###
        self.log("val_loss", loss, batch_size=batch_size)
        # return loss

    def training_step(self, batch, batch_idx):
        if self.trainer.global_step < self.warmup_steps:
            warmup_lr_schedule(self.trainer.optimizers[0], self.trainer.global_step, self.warmup_steps, self.warmup_lr, self.init_lr)
        loss = 0.0
        batch_size = batch[-1].size(0)
        ###============== Image-text Contrast ===================###
        if self.declip:
            graph, graph2, text, mask, text2, mask2 = batch

            batch_node_rep, batch_mask, batch_graph_rep = self.graph_encoder(graph)
            graph_rep = self.graph_proj_head(batch_graph_rep)

            batch_node_rep2, batch_mask2, batch_graph2_rep = self.graph_encoder(graph2)
            graph2_rep = self.graph_proj_head(batch_graph2_rep)

            text_rep = self.text_encoder(text, mask)
            text_rep = self.text_proj_head(text_rep)

            text2_rep = self.text_encoder(text2, mask2)
            text2_rep = self.text_proj_head(text2_rep)

            g2t_sim, t2g_sim, loss11 = self.contrast(graph_rep, text_rep)
            _, _, loss12 = self.contrast(graph_rep, text2_rep)
            _, _, loss21 = self.contrast(graph2_rep, text_rep)
            _, _, loss22 = self.contrast(graph2_rep, text2_rep)

            if self.graph_self:
                _, _, loss_graph_self = self.contrast(graph_rep, graph2_rep)
                loss_cl = (loss11 + loss12 + loss21 + loss22 + loss_graph_self) / 5.0
            else:
                loss_cl = (loss11 + loss12 + loss21 + loss22) / 4.0
        else:
            graph, text, mask = batch

            batch_node_rep, batch_mask, batch_graph_rep = self.graph_encoder(graph)
            graph_rep = self.graph_proj_head(batch_graph_rep)

            text_rep = self.text_encoder(text, mask)
            text_rep = self.text_proj_head(text_rep)

            g2t_sim, t2g_sim, loss_cl = self.contrast(graph_rep, text_rep)

        g_emb = torch.cat([batch_graph_rep.unsqueeze(1), batch_node_rep], dim=1)
        g_mask = torch.cat([torch.ones(batch_mask.size()[:-1], dtype=torch.long).unsqueeze(1).to(self.device), batch_mask], dim=1)

        ## use only 64 samples to calculate matching loss
        g_emb, g_mask = g_emb[:, :64, :], g_mask[:, :64]
        self.log("cl_loss", loss_cl, batch_size=batch_size)
        loss += loss_cl
        ###============== Image-text Matching ===================###
        if self.gtm:
            # bs = len(graph)
            encoder_input_ids = text.clone()
            encoder_input_ids[:, 0] = self.text_encoder.tokenizer.enc_token_id

            # forward the positve image-text pair
            output_pos = self.text_encoder(encoder_input_ids, mask, g_emb, g_mask)

            # select hard negatives based on sim matrix
            with torch.no_grad():
                weights_g2t = F.softmax(g2t_sim, dim=1) + 1e-4
                weights_g2t.fill_diagonal_(0)
                weights_t2g = F.softmax(t2g_sim, dim=1) + 1e-4
                weights_t2g.fill_diagonal_(0)

            # select a negative graph for each text
            graph_embeds_neg = []
            graph_mask_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_t2g[b], 1).item()
                graph_embeds_neg.append(g_emb[neg_idx])
                graph_mask_neg.append(g_mask[neg_idx])
            graph_embeds_neg = torch.stack(graph_embeds_neg, dim=0)
            graph_mask_neg = torch.stack(graph_mask_neg, dim=0)

            # select a negative text for each graph
            text_ids_neg = []
            text_atts_neg = []
            for b in range(batch_size):
                neg_idx = torch.multinomial(weights_g2t[b], 1).item()
                text_ids_neg.append(encoder_input_ids[neg_idx])
                text_atts_neg.append(mask[neg_idx])

            text_ids_neg = torch.stack(text_ids_neg, dim=0)
            text_atts_neg = torch.stack(text_atts_neg, dim=0)

            text_ids_all = torch.cat([encoder_input_ids, text_ids_neg], dim=0)
            text_atts_all = torch.cat([mask, text_atts_neg], dim=0)

            graph_embeds_all = torch.cat([graph_embeds_neg, g_emb], dim=0)
            graph_atts_all = torch.cat([graph_mask_neg, g_mask], dim=0)

            output_neg = self.text_encoder(text_ids_all,
                                           attention_mask=text_atts_all,
                                           g_emb=graph_embeds_all,
                                           g_mask=graph_atts_all,
                                           )
            gl_embeddings = torch.cat([output_pos, output_neg], dim=0)
            gl_output = self.gtm_head(gl_embeddings)
            gtm_labels = torch.cat([torch.ones(batch_size, dtype=torch.long), torch.zeros(2 * batch_size, dtype=torch.long)], dim=0).to(self.device)
            loss_gtm = F.cross_entropy(gl_output, gtm_labels)
            self.log("gtm_loss", loss_gtm, batch_size=batch_size)
            loss += loss_gtm

        ##================= LM ========================##
        if self.lm:
            decoder_input_ids = text.clone()
            decoder_input_ids[:, 0] = self.text_encoder.tokenizer.bos_token_id
            decoder_targets = decoder_input_ids.masked_fill(decoder_input_ids == self.text_encoder.tokenizer.pad_token_id, -100)

            decode_loss = self.text_encoder.decode_loss(decoder_input_ids,
                                                        attention_mask=mask,
                                                        g_emb=g_emb,
                                                        g_mask=g_mask,
                                                        labels=decoder_targets,
                                                        return_dict=True,
                                                        )

            loss_lm = decode_loss
            self.log("lm_loss", loss_lm, batch_size=batch_size)
            loss += loss_lm

        ###============== Overall Loss ===================###
        self.log("train_loss", loss, batch_size=batch_size)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size)
        return loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--graph_pooling', type=str, default='sum')
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--pretrain', type=str, default='scibert')
        parser.add_argument('--projection_dim', type=int, default=256)
        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-5, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=100, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        return parent_parser

