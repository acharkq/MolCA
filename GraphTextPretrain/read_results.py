import argparse
import pandas as pd
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    args = parser.parse_args()
    log_path = Path(args.path) / 'metrics.csv'
    log = pd.read_csv(log_path)
    print(log[~log['val_loss_gtc'].isnull()])
    
