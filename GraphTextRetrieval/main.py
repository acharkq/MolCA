import argparse
import random
import numpy as np
import torch
import util_funcs
from torch.autograd import Variable
from tqdm import tqdm
from model.blip2qformer import Blip2Qformer
from model.blip2_stage1 import Blip2Stage1
from data_provider.match_dataset import GINMatchDataset
import torch_geometric
from optimization import BertAdam
from pathlib import Path

import warnings
## for pyg bug
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

def prepare_model_and_optimizer(args, device):
    model = Blip2Stage1.load_from_checkpoint(args.init_checkpoint)
    model = model.blip2qformer
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
    assert isinstance(model, Blip2Qformer)
    model.eval()
    g2t_acc = 0
    t2g_acc = 0
    g2t_rec20 = 0
    t2g_rec20 = 0
    allcnt = 0
    graph_rep_total = []    
    text_rep_total = []
    for batch in tqdm(dataloader):
        aug, text, mask = batch
        aug = aug.to(device)
        text = text.to(device)
        mask = mask.to(device)
        graph_rep = model.graph_forward(aug) # shape = [B, num_qs, D]
        text_rep = model.text_forward(text, mask) # shape = [B, D]

        sim_q2t = (graph_rep.unsqueeze(1) @ text_rep.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [B, D, 1]; output shape = [B, B, num_qs]
        sim_g2t, _ = sim_q2t.max(-1) # shape = [B, B]

        B = sim_g2t.shape[0]
        sorted_ids = sim_g2t.argsort(descending=True).cpu()
        g2t_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = sim_g2t.T.argsort(descending=True).cpu()
        t2g_rank = (sorted_ids == torch.arange(B).reshape(-1, 1)).int().argmax(dim=-1)
        # argm1 = torch.argmax(sim_g2t, axis=1)
        # argm2 = torch.argmax(sim_g2t.T, axis=1)

        g2t_acc += float((g2t_rank == 0).sum())
        t2g_acc += float((t2g_rank == 0).sum())
        g2t_rec20 += float((g2t_rank < 20).sum())
        t2g_rec20 += float((t2g_rank < 20).sum())
        
        allcnt += B

        graph_rep_total.append(graph_rep.cpu())
        text_rep_total.append(text_rep.cpu())

    graph_rep_total = torch.cat(graph_rep_total, dim=0)
    text_rep_total = torch.cat(text_rep_total, dim=0)

    g2t_acc = round(g2t_acc/allcnt * 100, 2)
    t2g_acc = round(t2g_acc/allcnt * 100, 2)
    g2t_rec20 = round(g2t_rec20 / allcnt * 100, 2)
    t2g_rec20 = round(t2g_rec20 / allcnt * 100, 2)
    return g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep_total, text_rep_total


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
        g2t_acc, t2g_acc, g2t_rec20, t2g_rec20, graph_rep, text_rep = Eval(model, test_dataloader, device, args)
        print('G2T: Acc  Rec20  T2G: Acc   Rec20', )
        util_funcs.write_log(f'In batch:{args.init_checkpoint}  {g2t_acc}  {g2t_rec20}  {t2g_acc}  {t2g_rec20}', args.log_path)
        
        # graph_rep shape = [N, num_qs, D]; text_rep shape = [N, D]
        assert graph_rep.shape[0] == text_rep.shape[0] 
        N = graph_rep.shape[0]
        B = 8
        text_rep = text_rep.to(device)
        sim_g2t = []
        for i in tqdm(range(0, N, B)):
            l_graph_rep = graph_rep[i:i+B].to(device)
            l_sim_q2t = (l_graph_rep.unsqueeze(1) @ text_rep.unsqueeze(-1)).squeeze() # shape = [B, 1, num_qs, D]; shape = [N, D, 1]; output shape = [B, N, num_qs]
            l_sim_g2t, _ = l_sim_q2t.max(-1) # shape = [B, N]
            sim_g2t.append(l_sim_g2t)
        sim_g2t = torch.cat(sim_g2t, dim=0) # shape = [N, N]
        
        sorted_ids = torch.argsort(sim_g2t, descending=True)
        rank_g2t = (sorted_ids == torch.arange(N, device=device).reshape(-1, 1)).int().argmax(dim=-1)
        sorted_ids = torch.argsort(sim_g2t.T, descending=True)
        rank_t2g = (sorted_ids == torch.arange(N, device=device).reshape(-1, 1)).int().argmax(dim=-1)
        
        g2t_acc = float((rank_g2t == 0).float().mean())
        g2t_rec20 = float((rank_g2t < 20).float().mean())
        t2g_acc = float((rank_t2g == 0).float().mean())
        t2g_rec20 = float((rank_t2g < 20).float().mean())
        g2t_acc = round(g2t_acc * 100, 2)
        g2t_rec20 = round(g2t_rec20 * 100, 2)
        t2g_acc = round(t2g_acc * 100, 2)
        t2g_rec20 = round(t2g_rec20 * 100, 2)
        print('---------------------')
        print('G2T: Acc  Rec20  T2G: Acc   Rec20', )
        util_funcs.write_log(f'In test set:{args.init_checkpoint}  {g2t_acc}  {g2t_rec20}  {t2g_acc}  {t2g_rec20}', args.log_path)


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
    parser.add_argument("--init_checkpoint", default="all_checkpoints/cl_gtm_lm_50k/epoch=99-step=65700.ckpt", type=str)
    parser.add_argument("--run_type", default='zs', type=str, help='zs-zeroshot, ft-finetune')
    # parser.add_argument("--train_dataset", default='data/kv_data/train', type=str)
    # parser.add_argument("--val_dataset", default='data/kv_data/dev', type=str)
    # parser.add_argument("--test_dataset", default='our_data/PubChemDataset_v2/test', type=str)
    parser.add_argument("--test_dataset", default='data/PubChemDataset/PubChem-50k/test', type=str)
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
    args = parse_args()
    ckt_path = './all_checkpoints/cl_gtm_lm_50k'
    paths = list(Path(ckt_path).glob('*'))
    paths.sort()
    args.log_path = './logs/log.txt'
    for p in paths:
        args.init_checkpoint = str(p)
        main(args)
        torch.cuda.empty_cache()
