import argparse
import random
import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm
from model.contrastive_gin import GINSimclr
from data_provider.match_dataset import GINMatchDataset
import torch_geometric
from optimization import BertAdam
# from torch.utils.data import RandomSampler
# import os


def prepare_model_and_optimizer(args, device):
    model = GINSimclr.load_from_checkpoint(args.init_checkpoint)
    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [
                p for n, p in param_optimizer
                if not any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.01
        },
        {
            'params': [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            'weight_decay': 0.0
        },
    ]
    optimizer = BertAdam(optimizer_grouped_parameters,
                         weight_decay=args.weight_decay,
                         lr=args.lr,
                         warmup=args.warmup)

    return model, optimizer

@torch.no_grad()
def Eval(model, dataloader, device, args):
    model.eval()
    acc1 = 0
    acc2 = 0
    allcnt = 0
    graph_rep_total = None
    text_rep_total = None
    for batch in tqdm(dataloader):
        aug, text, mask = batch
        aug = aug.to(device)
        text = text.to(device)
        mask = mask.to(device)
        _, _, graph_rep = model.graph_encoder(aug)
        graph_rep = model.graph_proj_head(graph_rep)

        text_rep = model.text_encoder(text, mask)
        text_rep = model.text_proj_head(text_rep)

        scores1 = torch.cosine_similarity(graph_rep.unsqueeze(1).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), text_rep.unsqueeze(0).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), dim=-1)
        scores2 = torch.cosine_similarity(text_rep.unsqueeze(1).expand(text_rep.shape[0], text_rep.shape[0], text_rep.shape[1]), graph_rep.unsqueeze(0).expand(graph_rep.shape[0], graph_rep.shape[0], graph_rep.shape[1]), dim=-1)

        argm1 = torch.argmax(scores1, axis=1)
        argm2 = torch.argmax(scores2, axis=1)

        acc1 += sum((argm1 == torch.arange(argm1.shape[0]).to(device)).int()).item()
        acc2 += sum((argm2 == torch.arange(argm2.shape[0]).to(device)).int()).item()

        allcnt += argm1.shape[0]

        if graph_rep_total is None or text_rep_total is None:
            graph_rep_total = graph_rep
            text_rep_total = text_rep
        else:
            graph_rep_total = torch.cat((graph_rep_total, graph_rep), axis=0)
            text_rep_total = torch.cat((text_rep_total, text_rep), axis=0)

    return acc1/allcnt, acc2/allcnt, graph_rep_total.cpu(), text_rep_total.cpu()


def Contra_Loss(logits_des, logits_smi, margin, device):
    scores = torch.cosine_similarity(logits_smi.unsqueeze(1).expand(logits_smi.shape[0], logits_smi.shape[0], logits_smi.shape[1]), logits_des.unsqueeze(0).expand(logits_des.shape[0], logits_des.shape[0], logits_des.shape[1]), dim=-1)
    diagonal = scores.diag().view(logits_smi.size(0), 1)
    d1 = diagonal.expand_as(scores)
    d2 = diagonal.t().expand_as(scores)
    
    cost_des = (margin + scores - d1).clamp(min=0)
    cost_smi = (margin + scores - d2).clamp(min=0)

    mask = torch.eye(scores.size(0)) > .5
    I = Variable(mask)
    I = I.to(device)
    cost_des = cost_des.masked_fill_(I, 0)
    cost_smi = cost_smi.masked_fill_(I, 0)

    cost_des = cost_des.max(1)[0]
    cost_smi = cost_smi.max(0)[0]

    return cost_des.sum() + cost_smi.sum()


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
    model, optimizer = prepare_model_and_optimizer(args, device)

    TestSet = GINMatchDataset(args.test_dataset + '/', args)
    test_dataloader = torch_geometric.loader.DataLoader(TestSet, shuffle=True, batch_size=args.batch_size,
                                                        num_workers=4, pin_memory=True, drop_last=False)
    if args.run_type == 'zs':
        acc1, acc2, graph_rep, text_rep = Eval(model, test_dataloader, device, args)
        print('Test Acc G2T:', acc1)
        print('Test Acc T2G:', acc2)
        graph_len = graph_rep.shape[0]
        text_len = text_rep.shape[0]
        score1 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
        rec1 = []
        for i in range(graph_len):
            a, idx = torch.sort(score1[:, i])
            for j in range(graph_len):
                if idx[-1-j] == i:
                    rec1.append(j)
                    break
        print(f'Rec@20 G2T: {sum( (np.array(rec1)<20).astype(int) ) / graph_len}')
        score2 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
        rec2 = []
        for i in range(graph_len):
            a, idx = torch.sort(score2[:, i])
            for j in range(graph_len):
                if idx[-1-j] == i:
                    rec2.append(j)
                    break
        print(f'Rec@20 T2G: {sum( (np.array(rec2)<20).astype(int) ) / graph_len}')

    elif args.run_type == 'ft':
        TrainSet = GINMatchDataset(args.train_dataset + '/', args)
        ValSet = GINMatchDataset(args.val_dataset + '/', args)
        train_dataloader = torch_geometric.loader.DataLoader(TrainSet, shuffle=True, batch_size=args.batch_size,
                                                             num_workers=4, pin_memory=True, drop_last=True)
        val_dataloader = torch_geometric.loader.DataLoader(ValSet, shuffle=True, batch_size=args.batch_size,
                                                           num_workers=4, pin_memory=True, drop_last=False)
        best_val_acc = 0.0

        #Train
        for epoch in range(args.epoch):
            acc = 0
            allcnt = 0
            sumloss = 0
            model.train()
            for idx, batch in enumerate(tqdm(train_dataloader)):
                aug, text, mask = batch
                aug.to(device)
                text = text.to(device)
                mask = mask.to(device)

                _, _, graph_rep = model.graph_encoder(aug)
                graph_rep = model.graph_proj_head(graph_rep)

                text_rep = model.text_encoder(text, mask)
                text_rep = model.text_proj_head(text_rep)

                loss = Contra_Loss(graph_rep, text_rep, args.margin, device)
                scores = text_rep.mm(graph_rep.t())
                argm = torch.argmax(scores, axis=1)
                acc += sum((argm == torch.arange(argm.shape[0]).to(device)).int()).item()
                allcnt += argm.shape[0]
                sumloss += loss.item()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            print('Epoch:', epoch, ',Train Acc:', acc/allcnt, ', Train Loss:', sumloss/allcnt)

            # Eval
            val_acc1, val_acc2, _, _ = Eval(model, val_dataloader, device, args)
            print('Epoch:', epoch, ', Val Acc G2T:', val_acc1, ', ValAcc T2G:', val_acc2)

            test_acc1, test_acc2, graph_rep, text_rep = Eval(model, test_dataloader, device, args)
            print('Epoch:', epoch, ', Test Acc G2T:', test_acc1, ', Test Acc T2G:', test_acc2)

            if val_acc1 > best_val_acc:
                best_val_acc = val_acc1
                best_g2t_acc = test_acc1
                best_t2g_acc = test_acc2
                best_graph_rep = graph_rep
                best_text_rep = text_rep

        graph_rep, text_rep = best_graph_rep, best_text_rep
        graph_len = graph_rep.shape[0]
        text_len = text_rep.shape[0]
        score1 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score1[i] = torch.cosine_similarity(graph_rep[i], text_rep, dim=-1)
        rec1 = []
        for i in range(graph_len):
            a, idx = torch.sort(score1[:, i])
            for j in range(graph_len):
                if idx[-1 - j] == i:
                    rec1.append(j)
                    break
        rec_g2t = sum((np.array(rec1) < 20).astype(int)) / graph_len
        # print(f'Rec@20 G2T: {sum((np.array(rec1) < 20).astype(int)) / graph_len}')
        score2 = torch.zeros(graph_len, graph_len)
        for i in range(graph_len):
            score2[i] = torch.cosine_similarity(text_rep[i], graph_rep, dim=-1)
        rec2 = []
        for i in range(graph_len):
            a, idx = torch.sort(score2[:, i])
            for j in range(graph_len):
                if idx[-1 - j] == i:
                    rec2.append(j)
                    break
        rec_t2g = sum((np.array(rec2) < 20).astype(int)) / graph_len
        # print(f'Rec@20 T2G: {sum((np.array(rec2) < 20).astype(int)) / graph_len}')
        print('\n Final Test Acc G2T:', best_g2t_acc, ', Test Acc T2G:', best_t2g_acc,
              'Test Rec@20 G2T:', rec_g2t, ', Test Rec@20 T2G:', rec_t2g)


def parse_args(parser=argparse.ArgumentParser()):
    parser.add_argument("--device", default="0", type=str)
    parser.add_argument("--init_checkpoint", default="all_checkpoints/cl_gtm_lm_320k_v2/epoch=29.ckpt", type=str)
    parser.add_argument("--run_type", default='zs', type=str, help='zs-zeroshot, ft-finetune')
    # parser.add_argument("--train_dataset", default='data/kv_data/train', type=str)
    # parser.add_argument("--val_dataset", default='data/kv_data/dev', type=str)
    # parser.add_argument("--test_dataset", default='our_data/PubChemDataset_v2/test', type=str)
    parser.add_argument("--test_dataset", default='data/phy_data', type=str)
    parser.add_argument("--weight_decay", default=0, type=float)
    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--warmup", default=0.2, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epoch", default=30, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--graph_aug", default='noaug', type=str)
    parser.add_argument("--text_max_len", default=128, type=int)
    parser.add_argument("--margin", default=0.2, type=int)
    
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    main(parse_args())
