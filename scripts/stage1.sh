{

python stage1.py --root 'data/PubChemDataset_v4' --gtm --lm --devices '0,1' --mode train --filename stage1_q32_tune_gnn --rerank_cand_num 128 --num_query_token 8 --tune_gnn

# python stage1.py --root 'data/kv_data' --gtm --lm --devices '4,5'  --filename pcdes --init_checkpoint "all_checkpoints/stage1_default_tune_gnn/epoch=49-step=120950.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval --use_phy_eval

exit
}