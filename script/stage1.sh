{

python stage1.py --root 'data/PubChem324k' --gtm --lm --devices '6,7' --mode train --filename stage1 --rerank_cand_num 128 --num_query_token 8 --tune_gnn --filtered_cid_path data/PubChem324k/filtered_pretrain_cids.txt

# python stage1.py --root 'data/PubChem324k' --gtm --lm --devices '1,5' --mode train --filename stage1 --rerank_cand_num 128 --num_query_token 8 --tune_gnn --filtered_cid_path data/PubChem324k/filtered_pretrain_cids.txt --retrieval_eval_epoch 1
exit
}