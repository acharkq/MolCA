import os
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from data_provider.pretrain_stage2_datamodule import PretrainStage2DataModule
from model.blip2_stage2 import Blip2Stage2
import warnings
from pytorch_lightning import strategies
from pytorch_lightning.loggers import CSVLogger

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)
    # model
    if args.init_checkpoint:
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False)
        print(f"loaded stage2 model from {args.init_checkpoint}")
    else:
        model = Blip2Stage2(args)
        # model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
        
    print('total params:', sum(p.numel() for p in model.parameters()))

    # data
    # dm = GINPretrainDataModule.from_argparse_args(args)
    dm = PretrainStage2DataModule(args.num_workers, args.batch_size, args.root, args.text_max_len, args)
    dm.tokenizer = model.blip2opt.opt_tokenizer

    callbacks = []
    ## fixme save only used parameters
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    
    strategy = strategies.DDPSpawnStrategy(find_unused_parameters=False)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                        #  limit_train_batches=20,
                                        #  limit_val_batches=5,
                                         )
    
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 9 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='train')
    parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    parser = PretrainStage2DataModule.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision=16,
                        max_epochs=10,
                        accumulate_grad_batches=1,
                        check_val_every_n_epoch=1)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

