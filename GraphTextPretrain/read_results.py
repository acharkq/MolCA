import argparse
import pandas as pd
from pathlib import Path

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--disable_rerank', action='store_true', default=False)
    args = parser.parse_args()
    args.path = Path(args.path)
    log_hparas = args.path / 'hparams.yaml'
    with open(log_hparas, 'r') as f:
        line = f.readline()
        file_name = line.strip().split(' ')[1]
    
    log_path = args.path / 'metrics.csv'
    log = pd.read_csv(log_path)
    log = log.round(2)
    print(f'File name: {file_name}')
    if args.disable_rerank:
        retrieval_cols = ['test_inbatch_g2t_acc', 'test_inbatch_g2t_rec20', 'test_inbatch_t2g_acc',   'test_inbatch_t2g_rec20',  'test_fullset_g2t_acc',  'test_fullset_g2t_rec20', 'test_fullset_t2g_acc', 'test_fullset_t2g_rec20']
    else:
        retrieval_cols = ['rerank_test_inbatch_g2t_acc', 'rerank_test_inbatch_g2t_rec20', 'rerank_test_inbatch_t2g_acc',   'rerank_test_inbatch_t2g_rec20', 'rerank_test_fullset_g2t_acc',  'rerank_test_fullset_g2t_rec20', 'rerank_test_fullset_t2g_acc', 'rerank_test_fullset_t2g_rec20']
    retrieval_log = log[~log['test_inbatch_t2g_acc'].isnull()][retrieval_cols]
    print(retrieval_cols)
    print(retrieval_log.to_string(header=False))
    
    
