import argparse
import random
import numpy as np
import torch
from tqdm import tqdm
from model.bert import MM_Decoder
from model.contrastive_gin import GINSimclr
from data_provider.match_dataset import CapDataset
import torch_geometric


@torch.no_grad()
def Eval(model, dataloader, device, args):
    model.eval()
    for batch in tqdm(dataloader):
        captions = model.generate(batch)
        exit()


def main(args):
    print("=========================================")
    for k, v in sorted(vars(args).items()):
        print(k, '=', v)
    print("=========================================")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = torch.device("cuda:" + args.device if torch.cuda.is_available() else "cpu")
    # model = MM_Decoder()
    # model.load_checkpoint(args.init_checkpoint)
    model = GINSimclr.load_from_checkpoint(args.init_checkpoint)
    model.to(device)

    TestSet = CapDataset(args.test_dataset + '/', args)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=True, batch_size=args.batch_size,
                                                        num_workers=0, pin_memory=True, drop_last=False)
    if args.run_type == 'zs':
        Eval(model, test_dataloader, device, args)


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--device", default="5", type=str)
    parser.add_argument("--init_checkpoint", default="all_checkpoints/cl_gtm_lm_320k_v2/epoch=29.ckpt", type=str)
    parser.add_argument("--test_dataset", default='our_data/PubChemDataset_v2/train', type=str)
    parser.add_argument("--run_type", default='zs', type=str, help='zs-zeroshot, ft-finetune')
    # parser.add_argument("--weight_decay", default=0, type=float)
    # parser.add_argument("--lr", default=5e-5, type=float)
    # parser.add_argument("--warmup", default=0.2, type=float)
    parser.add_argument("--batch_size", default=16, type=int)
    # parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    # parser.add_argument("--graph_aug", default='noaug', type=str)
    parser.add_argument("--text_max_len", default=256, type=int)
    # parser.add_argument("--margin", default=0.2, type=int)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
