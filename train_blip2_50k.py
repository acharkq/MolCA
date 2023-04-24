import os
import argparse
import torch
import random
import numpy as np
from torch import optim
import torch.nn as nn
import time
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model.blip2_stage1 import Blip2Stage1
from data_provider.pretrain_datamodule import GINPretrainDataModule
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def main(args):
    pl.seed_everything(args.seed)

    # model
    model = Blip2Stage1(args)
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    dm = GINPretrainDataModule.from_argparse_args(args)
    dm.train_dataset.tokenizer = model.blip2qformer.tokenizer
    dm.val_dataset.tokenizer = model.blip2qformer.tokenizer
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

    parser.add_argument('--filename', type=str, default="cl_gtm_lm_50k")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_true', help='use graph-text matching or not', default=True)

    parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule.add_argparse_args(parser)  # add data args

    parser.set_defaults(batch_size=16,
                        accelerator='gpu',
                        gpus='0,1,2,3',
                        precision=16,
                        max_epochs=100,
                        num_workers=8,
                        declip=True,
                        root='data/PubChemDataset/PubChem-50k',
                        check_val_every_n_epoch=1,
                        )
    args = parser.parse_args()
    
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    ## for A5000 gpus
    torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)
    main(args)

