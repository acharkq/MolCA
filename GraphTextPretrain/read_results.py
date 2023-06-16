import argparse
import pandas as pd
from pathlib import Path
import json
from model.blip2_stage2 import caption_evaluate
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW, 
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)
from transformers import AutoTokenizer, BertTokenizer

pd.options.display.max_rows = 1000
pd.options.display.max_columns = 1000


def get_mode(df):
    if 'bleu2' in df.columns:
        return 'caption'
    elif 'test_inbatch_g2t_acc' in df.columns:
        return 'retrieval'
    else:
        raise NotImplementedError


def read_retrieval(df, args):
    df = df.round(2)
    if args.disable_rerank:
        retrieval_cols = ['test_inbatch_g2t_acc', 'test_inbatch_g2t_rec20', 'test_inbatch_t2g_acc',   'test_inbatch_t2g_rec20',  'test_fullset_g2t_acc',  'test_fullset_g2t_rec20', 'test_fullset_t2g_acc', 'test_fullset_t2g_rec20']
    else:
        retrieval_cols = ['rerank_test_inbatch_g2t_acc', 'rerank_test_inbatch_g2t_rec20', 'rerank_test_inbatch_t2g_acc',   'rerank_test_inbatch_t2g_rec20', 'rerank_test_fullset_g2t_acc',  'rerank_test_fullset_g2t_rec20', 'rerank_test_fullset_t2g_acc', 'rerank_test_fullset_t2g_rec20']
    retrieval_log = df[~df['test_inbatch_t2g_acc'].isnull()][retrieval_cols]
    print(retrieval_cols)
    print(retrieval_log.to_string(header=False))

def read_caption(df, args):
    df = df.round(2)
    df = df[~df['bleu2'].isnull()]
    cols = ['epoch', 'bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
    caption_log = df[cols]
    
    print(cols)
    print(caption_log)

def exact_match(prediction_list, target_list):
    match = 0
    for prediction, target in zip(prediction_list, target_list):
        prediction = prediction.strip()
        target = target.strip()
        if prediction == target:
            match += 1
    acc = round(match / len(prediction_list) * 100, 2)
    return acc

def read_caption_prediction(args):
    path = args.path
    with open(path, 'r') as f:
        lines = f.readlines()
        lines = [json.loads(line) for line in lines]
    # tokenizer = AutoTokenizer.from_pretrained('./bert_pretrained')
    tokenizer = BertTokenizer.from_pretrained('bert_pretrained/')
    tokenizer.add_special_tokens({"bos_token": "[DEC]"})
        

    prediction_list = []
    target_list = []
    for line in lines:
        prediction = line['prediction'].strip()
        target = line['target'].strip()
        prediction_list.append(prediction)
        target_list.append(target)
    bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score = caption_evaluate(prediction_list, target_list, tokenizer, 512)
    bleu2 = round(bleu2, 2)
    bleu4 = round(bleu4, 2)
    rouge_1 = round(rouge_1, 2)
    rouge_2 = round(rouge_2, 2)
    rouge_l = round(rouge_l, 2)
    meteor_score = round(meteor_score, 2)
    if str(args.path).find('iupac') >= 0:
        acc = exact_match(prediction_list, target_list)
        cols = ['Exact match', 'bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
        print(cols)
        print(acc, bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score)
    else:
        cols = ['bleu2','bleu4','rouge_1','rouge_2','rouge_l','meteor_score']
        print(cols)
        print(bleu2, bleu4, rouge_1, rouge_2, rouge_l, meteor_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--tag', type=str, default='train_loss_gtm')
    parser.add_argument('--max_step', type=int, default=1000)
    parser.add_argument('--disable_rerank', action='store_true', default=False)
    args = parser.parse_args()
    args.path = Path(args.path)
    
    if args.path.name == 'predictions.txt':
        read_caption_prediction(args)
        exit()
    
    log_hparas = args.path / 'hparams.yaml'
    with open(log_hparas, 'r') as f:
        line = f.readline()
        file_name = line.strip().split(' ')[1]
    
    log_path = args.path / 'metrics.csv'
    log = pd.read_csv(log_path)
    
    print(f'File name: {file_name}')
    mode = get_mode(log)
    
    if mode == 'retrieval':
        read_retrieval(log, args)
    elif mode == 'caption':
        read_caption(log, args)
    