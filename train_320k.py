import os
import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import time
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import TensorBoardLogger
from model.contrastive_gin import GINSimclr
from torch_geometric.data import LightningDataset
from data_provider.pretrain_datamodule import GINPretrainDataModule
from data_provider.pretrain_dataset import GINPretrainDataset


def main(args):
    pl.seed_everything(args.seed)

    # model
    model = GINSimclr(
        temperature=args.temperature,
        gin_hidden_dim=args.gin_hidden_dim,
        gin_num_layers=args.gin_num_layers,
        drop_ratio=args.drop_ratio,
        graph_pooling=args.graph_pooling,
        graph_self=args.graph_self,
        declip=args.declip,
        gtm=args.gtm,
        lm=args.lm,
        bert_hidden_dim=args.bert_hidden_dim,
        pretrain=args.pretrain,
        projection_dim=args.projection_dim,
        weight_decay=args.weight_decay,
        init_lr=args.init_lr,
        min_lr=args.min_lr,
        warmup_lr=args.warmup_lr,
        warmup_steps=args.warmup_steps,
        lr_decay_rate=args.lr_decay_rate,
    )
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = GINPretrainDataModule.from_argparse_args(args)
    dm.train_dataset.tokenizer = model.text_encoder.tokenizer
    dm.val_dataset.tokenizer = model.text_encoder.tokenizer
    # tokenizer syc

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    strategy = pl.strategies.DDPSpawnStrategy(find_unused_parameters=False)
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         # accumulate_grad_batches=8,
                                         )

    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="cl_gtm_lm_320k")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--graph_self', action='store_true', help='use graph self-supervise or not', default=True)

    parser = Trainer.add_argparse_args(parser)
    parser = GINSimclr.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args

    parser.set_defaults(batch_size=48,
                        accelerator='gpu',
                        gpus='0,1,2,3',
                        # precision=16,
                        max_epochs=50,
                        num_workers=8,
                        declip=True,
                        root='data/PubChemDataset/PubChem-320k',
                        check_val_every_n_epoch=1,
                        warmup_steps=1000
                        )
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")

    main(args)








