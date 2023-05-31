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
        model = Blip2Stage2.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = Blip2Stage2(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage1_path}")
    elif args.stage1_path:
        model = Blip2Stage2(args)
        model.load_from_stage1_checkpoint(args.stage1_path)
        print(f"loaded stage1 model from {args.stage1_path}")
    else:
        model = Blip2Stage2(args)

    print('total params:', sum(p.numel() for p in model.parameters()))

    if args.opt_model.find('opt') >= 0:
        tokenizer = model.blip2opt.opt_tokenizer
    elif args.opt_model.find('llama') >= 0 or args.opt_model.find('vicuna') >= 0:
        tokenizer = model.blip2opt.llm_tokenizer
    else:
        raise NotImplementedError
    # data
    dm = PretrainStage2DataModule(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, tokenizer, args)

    callbacks = []
    ## fixme save only used parameters
    # callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", every_n_epochs=10, save_top_k=-1))
    callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                         filename='{epoch:02d}', 
                                         every_n_epochs=10, 
                                         save_last=True, 
                                         save_top_k=-1))
    
    # strategy = strategies.DDPStrategy(find_unused_parameters=False)
    args.devices = eval(args.devices)
    if len(args.devices) == 1:
        strategy = None
    else:
        strategy = strategies.DDPSpawnStrategy(find_unused_parameters=False)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=strategy,
                                         logger=logger,
                                        #  limit_train_batches=20,
                                        #  limit_test_batches=100,
                                         )
    
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm)
        trainer.test(model, datamodule=dm)
    elif args.mode == 'eval':
        trainer.test(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='pretrain')
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
    return args

if __name__ == '__main__':
    main(get_args())

