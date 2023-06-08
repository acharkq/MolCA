{

python stage1.py --root 'data/PubChemDataset_v4' --gtm --lm --devices '4,5'  --filename stage1_q32 --rerank_cand_num 128 --num_query_token 32 --match_batch_size 16

exit
}