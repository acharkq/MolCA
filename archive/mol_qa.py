import os
import torch
import argparse
import warnings
import pytorch_lightning as pl
from pytorch_lightning import Trainer, strategies
import pytorch_lightning.callbacks as plc
from pytorch_lightning.loggers import CSVLogger
from model.blip2_opt import Blip2OPT
from model.blip2_stage2 import Blip2Stage2
# from data_provider.stage2_dm import Stage2DM

os.environ['OPENBLAS_NUM_THREADS'] = '1'
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')
## for A5000 gpus
torch.set_float32_matmul_precision('medium') # can be medium (bfloat16), high (tensorfloat32), highest (float32)


def main(args):
    pl.seed_everything(args.seed)
    # model
    model= Blip2OPT(args.bert_name, args.gin_num_layers, args.gin_hidden_dim, args.drop_ratio, args.tune_gnn, args.num_query_token, args.cross_attention_freq, args.llm_tune, args.peft_dir, args.opt_model, args.prompt)
    if args.init_checkpoint:
        state_dict = torch.load(args.init_checkpoint, map_location='cpu')['state_dict']
        state_dict = {k[9:]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=False)
        print(f"loaded init checkpoint from {args.init_checkpoint}")
    
    assert len(args.device) == 1
    model.to(f'cuda:{args.device}')
    print('total params:', sum(p.numel() for p in model.parameters()))
    while True:
        question = input('input your question: \n')
        samples = {'prompts': [question]}
        try:
            output = model.blip_qa(
                samples, 
                do_sample=True,
                num_beams=5,
                max_length=128,
                min_length=8,
                temperature=0.7,
            )
        except RuntimeError as e:
            print(e)
            output = 'blank'
        print('The answer is:')
        print(output)
        print('\n-----------------')
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str, default="stage2_test")
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    # MM settings
    
    parser.add_argument('--mode', type=str, default='pretrain')
    parser.add_argument('--device', type=str, default='0')
    parser.add_argument('--prompt', type=str, default='The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ')
    # parser = Trainer.add_argparse_args(parser)
    parser = Blip2Stage2.add_model_specific_args(parser)  # add model args
    # parser = Stage2DM.add_model_specific_args(parser)
    parser.set_defaults(accelerator='gpu',
                        devices='0,1,2,3',
                        precision='bf16',
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

