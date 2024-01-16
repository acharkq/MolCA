import os
import argparse
import torch
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_stage1 import Blip2Stage1
from data_provider.stage1_dm import Stage1DM
from data_provider.stage1_kvplm_dm import Stage1KVPLMDM


os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)

    # model
    if args.init_checkpoint:
        model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint, device=args.devices, args=args)
        print(f"loading model from {args.init_checkpoint}")
    else:
        model = Blip2Stage1(args)
    
    print('total params:', sum(p.numel() for p in model.parameters()))

    tokenizer = model.blip2qformer.tokenizer
    # data
    if args.root.find('kv') >= 0:
        dm = Stage1KVPLMDM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, args)
    else:
        dm = Stage1DM(args.num_workers, args.batch_size, args.root, args.text_max_len, args.graph_aug, tokenizer,
                      args)
    model.val_match_loader = dm.val_match_loader
    model.test_match_loader = dm.test_match_loader

    callbacks = []
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=args.save_every_n_epochs, 
                                         save_top_k=-1))
    
    find_unused_parameters = (not args.gtm) or (not args.lm)
    if len(args.devices.split(',')) > 1:
        strategy = strategies.DDPStrategy(find_unused_parameters=find_unused_parameters, start_method='spawn')
    else:
        strategy = 'auto'
        args.devices = eval(args.devices)
        print(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    # trainer = Trainer.from_argparse_args(args,
    #                                      callbacks=callbacks,
    #                                      strategy=strategy,
    #                                      logger=logger,
    #                                     #  limit_train_batches=100,
    #                                      )
    trainer = Trainer(accelerator=args.accelerator, devices=args.devices, precision=args.precision, max_epochs=args.max_epochs, check_val_every_n_epoch=args.check_val_every_n_epoch, callbacks=callbacks, strategy=strategy, logger=logger)
    if args.mode == 'train':
        trainer.fit(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = 49 ## avoid 
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--filename', type=str, default="stage1_test")
    # GPU
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--gtm', action='store_true', help='use graph-text matching or not', default=True)
    parser.add_argument('--lm', action='store_true', help='use language modeling or not', default=True)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--accelerator', type=str, default='gpu')
    parser.add_argument('--devices', type=str, default='0,1,2,3')
    parser.add_argument('--precision', type=str, default='bf16-mixed')
    parser.add_argument('--max_epochs', type=int, default=50)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    # parser.add_argument('--save_every_n_epochs', type=int, default=1)
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage1.add_model_specific_args(parser)  # add model args
    parser = Stage1DM.add_model_specific_args(parser)
    # parser.set_defaults(accelerator='gpu',
    #                     devices='0,1,2,3',
    #                     precision='bf16',
    #                     max_epochs=50,
    #                     check_val_every_n_epoch=1)
    args = parser.parse_args()
    
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    main(args)

