{


python stage1.py --root 'data/PubChem324k' --gtm --lm --devices '0,1' --mode train --filename stage1 --rerank_cand_num 128 --num_query_token 8 --tune_gnn --filtered_cid_path data/PubChemDataset_v4/filtered_pretrain_cids.txt
}