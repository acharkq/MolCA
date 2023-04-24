import torch
from model.blip2qformer import Blip2Qformer
import pytorch_lightning as pl
from torch import optim
from lavis.common.optims import LinearWarmupCosineLRScheduler, LinearWarmupStepLRScheduler


# def warmup_lr_schedule(optimizer, step, max_step, init_lr, max_lr):
#     """Warmup the learning rate"""
#     lr = min(max_lr, init_lr + (max_lr - init_lr) * step / max_step)
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


# def step_lr_schedule(optimizer, epoch, init_lr, min_lr, decay_rate):
#     """Decay the learning rate"""
#     lr = max(min_lr, init_lr * (decay_rate**epoch))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr


class Blip2Stage1(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        # self.declip = declip
        # self.gtm = gtm
        # self.lm = lm
        # self.init_lr = init_lr
        # self.min_lr = min_lr
        # self.warmup_lr = warmup_lr
        # self.warmup_steps = warmup_steps
        # self.lr_decay_rate = lr_decay_rate

        
        # self.temperature = temperature
        # self.gin_hidden_dim = gin_hidden_dim
        # self.gin_num_layers = gin_num_layers
        # self.drop_ratio = drop_ratio

        # self.bert_name = bert_name

        # self.projection_dim = projection_dim
        # self.weight_decay = weight_decay
        self.blip2qformer = Blip2Qformer(args.gtm, args.lm, args.bert_name, args.declip, args.temperature, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.freeze_gnn, args.num_query_token, args.cross_attention_freq, args.projection_dim)
    
        self.save_hyperparameters(args)
        

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.args.init_lr, weight_decay=self.args.weight_decay)
        if self.args.scheduler == 'linear_warmup_cosine_lr':
            self.scheduler = LinearWarmupCosineLRScheduler(self.trainer.optimizers[0], self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.warmup_steps, self.args.warmup_lr)
        elif self.args.scheduler == 'linear_warmup_step_lr':
            self.scheduler = LinearWarmupStepLRScheduler(self.trainer.optimizers[0], self.args.max_epochs, self.args.min_lr, self.args.init_lr, self.args.lr_decay_rate, self.args.warmup_lr, self.args.warmup_steps)
        else:
            raise NotImplementedError()
        return optimizer

    # def on_train_epoch_start(self) -> None:
    #     step_lr_schedule(self.trainer.optimizers[0], self.trainer.current_epoch, self.args.init_lr, self.args.min_lr, self.args.lr_decay_rate)

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        batch_size = batch[-1].size(0)
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("val_loss_gtc", blip2_loss.loss_itc.item(), batch_size=batch_size)
        self.log("val_loss_gtm", blip2_loss.loss_itm.item(), batch_size=batch_size)
        self.log("val_loss_lm", blip2_loss.loss_lm.item(), batch_size=batch_size)
        self.log("val_loss", blip2_loss.loss.item(), batch_size=batch_size)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size)
        return blip2_loss.loss
    
    def training_step(self, batch, batch_idx):
        # if self.trainer.global_step < self.args.warmup_steps:
        #     warmup_lr_schedule(self.trainer.optimizers[0], self.trainer.global_step, self.args.warmup_steps, self.args.warmup_lr, self.args.init_lr)
        self.scheduler.step(self.trainer.current_epoch, self.trainer.global_step)

        batch_size = batch[-1].size(0)
        blip2_loss = self.blip2qformer(batch)
        ###============== Overall Loss ===================###
        self.log("train_loss_gtc", blip2_loss.loss_itc.item(), batch_size=batch_size)
        self.log("train_loss_gtm", blip2_loss.loss_itm.item(), batch_size=batch_size)
        self.log("train_loss_lm", blip2_loss.loss_lm.item(), batch_size=batch_size)
        self.log("train_loss", blip2_loss.loss.item(), batch_size=batch_size)
        self.log("lr", self.trainer.optimizers[0].param_groups[0]['lr'], batch_size=batch_size)
        return blip2_loss.loss

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("GINSimclr")
        # train mode
        parser.add_argument('--temperature', type=float, default=0.1, help='the temperature of NT_XentLoss')
        # GIN
        parser.add_argument('--gin_hidden_dim', type=int, default=300)
        parser.add_argument('--gin_num_layers', type=int, default=5)
        parser.add_argument('--drop_ratio', type=float, default=0.0)
        parser.add_argument('--freeze_gnn', action='store_true', default=True)
        # Bert
        parser.add_argument('--bert_hidden_dim', type=int, default=768, help='')
        parser.add_argument('--bert_name', type=str, default='scibert')
        parser.add_argument('--projection_dim', type=int, default=256)
        parser.add_argument('--cross_attention_freq', type=int, default=2)
        parser.add_argument('--num_query_token', type=int, default=32)
        # optimization
        parser.add_argument('--weight_decay', type=float, default=0.05, help='optimizer weight decay')
        parser.add_argument('--init_lr', type=float, default=1e-4, help='optimizer init learning rate')
        parser.add_argument('--min_lr', type=float, default=1e-5, help='optimizer min learning rate')
        parser.add_argument('--warmup_lr', type=float, default=1e-5, help='optimizer warmup learning rate')
        parser.add_argument('--warmup_steps', type=int, default=100, help='optimizer warmup steps')
        parser.add_argument('--lr_decay_rate', type=float, default=0.9, help='optimizer lr decay rate')
        parser.add_argument('--scheduler', type=str, default='linear_warmup_step_lr', help='type of scheduler') # or linear_warmup_cosine_lr
        return parent_parser
