import argparse
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from model.blip2_stage1 import Blip2Stage1
from data_provider.pretrain_datamodule import GINPretrainDataModule
from data_provider.pretrain_datamodule_v2 import GINPretrainDataModule_v2
import warnings
from pytorch_lightning import strategies
import os
from pytorch_lightning.loggers import CSVLogger

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
# torch.set_float32_matmul_precision('high') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    # dm = GINPretrainDataModule.from_argparse_args(args)
    dm = GINPretrainDataModule_v2(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, args.declip, args)
    dm.train_dataset.tokenizer = model.blip2qformer.tokenizer
    dm.val_dataset.tokenizer = model.blip2qformer.tokenizer
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    
    find_unused_parameters = (not args.gtm) or (not args.lm)
    strategy = strategies.DDPSpawnStrategy(find_unused_parameters=find_unused_parameters)
    logger = CSVLogger(save_dir='./')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                        #  limit_train_batches=100,
                                         )
    
    trainer.fit(model, datamodule=dm)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="cl_gtm_lm_50k")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=False)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=False)
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = GINPretrainDataModule_v2.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision=32,
                        max_epochs=50,
                        check_val_every_n_epoch=1)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

