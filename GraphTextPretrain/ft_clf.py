import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from data_provider.clf_dm import FGClfDM
from data_provider.mpp_dm import MPPDM
from model.molca_ft_clf import MolCAClf
from model.smiles_opt_clf import SmilesClf


# torch.set_default_dtype(torch.float16)

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
# torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)
    if args.root.find('MoleculeNet') >= 0:
        dm = MPPDM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, None, args)
    else:
        dm = FGClfDM(args.mode, args.num_workers, args.batch_size, args.root, args.text_max_len, None, args)
        args.num_labels = dm.train_dataset.num_labels

    # dm.train_dataset.num_label
    # 

    if args.smiles_only:
        print('running smiles model')
        model_cls = SmilesClf
    else:
        model_cls = MolCAClf

    # model
    if args.init_checkpoint:
        model = model_cls.load_from_checkpoint(args.init_checkpoint, strict=False, args=args)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    elif args.stage2_path:
        model = model_cls(args)
        ckpt = torch.load(args.stage2_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'], strict=False)
        print(f"loaded stage2 model from {args.stage2_path}")
    else:
        model = model_cls(args)
    print('total params:', sum(p.numel() for p in model.parameters()))
    
    if args.smiles_only:
        tokenizer = model.tokenizer
    else:
        tokenizer = model.blip2opt.opt_tokenizer
    
    dm.init_tokenizer(tokenizer)

    callbacks = []
    if args.root.find('MoleculeNet') >= 0:
        ## fixme save only used parameters
        callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                            filename='{epoch:02d}', 
                                            every_n_epochs=0, 
                                            save_last=False, 
                                            save_top_k=0))
    else:
        callbacks.append(plc.ModelCheckpoint(dirpath="all_checkpoints/"+args.filename+"/", 
                                            filename='{epoch:02d}', 
                                            every_n_epochs=args.save_every_n_epochs, 
                                            save_last=False, 
                                            save_top_k=-1))
        
    if not isinstance(args.devices, list):
        args.devices = eval(args.devices)
    logger = CSVLogger(save_dir=f'./all_checkpoints/{args.filename}/')
    trainer = Trainer.from_argparse_args(args,
                                         callbacks=callbacks,
                                         strategy=None,
                                         logger=logger,
                                        #  limit_train_batches=20,
                                        #  limit_test_batches=100,
                                         )
    
    if args.mode in {'pretrain', 'ft'}:
        trainer.fit(model, datamodule=dm, ckpt_path=args.ckpt_path)
    elif args.mode == 'eval':
        trainer.fit_loop.epoch_progress.current.completed = args.caption_eval_epoch - 1
        trainer.validate(model, datamodule=dm)
    else:
        raise NotImplementedError()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    parser.add_argument('--use_bn', action='store_true', default=False)
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--strategy_name', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--smiles_only', action='store_true', default=False)
    parser.add_argument('--llm_name', type=str, default='facebook/galactica-1.3b')
    parser = Trainer.add_argparse_args(parser)
    parser = MolCAClf.add_model_specific_args(parser)  # add model args
    parser = MPPDM.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0',
                        precision=16,
                        max_epochs=10,
                        accumulate_grad_batches=1,
                        log_every_n_steps=5,
                        check_val_every_n_epoch=1)
    args = parser.parse_args()

    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    return args


def run_moleculenet(args):
    dataset_list = ['bace', 'bbbp', 'clintox', 'toxcast', 'sider', 'tox21']
    # dataset_list = ['sider', 'tox21']
    num_task_dict = {'tox21': 12, 'hiv': 1, 'muv': 17, 'bace': 1,
                         'bbbp': 1, 'toxcast': 617, 'sider': 27, 'clintox': 2}
    print('running moleculenet')
    args.max_epochs = 100
    filename = args.filename
    for dataset in dataset_list:
        args.filename = f"{filename}/{dataset}"
        args.num_labels = num_task_dict[dataset]
        print(args.filename)
        for seed in range(0, 3):
            args.seed = seed
            args.tuning_dataset = dataset
            main(args)

if __name__ == '__main__':
    args = get_args()
    if args.root.find('MoleculeNet') >= 0:
        run_moleculenet(args)
    else:
        main(args)
    
