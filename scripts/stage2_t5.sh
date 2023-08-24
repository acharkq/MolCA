{
export TOKENIZERS_PARALLELISM=false;
python stage2.py --devices '0,1,2,3' --filename "molt5_stage2" --stage1_path "all_checkpoints/share/stage1.ckpt" --opt_model 'laituan245/molt5-large' --mode pretrain --prompt '[START_SMILES]{}[END_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8 --root "data/PubChemDataset_v4" --batch_size 16 --accumulate_grad_batches 2 --peft_config scripts/lora_config_t5.json; 

exit
}