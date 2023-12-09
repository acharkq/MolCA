{
export TOKENIZERS_PARALLELISM=false;
# python stage2.py --root 'data/PubChem324k' --devices '0,1' --filename "stage2" --stage1_path "all_checkpoints/stage1/last.ckpt" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune freeze --inference_batch_size 4

python stage2.py --root 'data/PubChem324k' --devices '1,5' --filename "stage2" --opt_model 'facebook/galactica-1.3b' --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune freeze --inference_batch_size 4 --caption_eval_epoch 1 --filtered_cid_path data/PubChem324k/filtered_pretrain_cids.txt
exit
}