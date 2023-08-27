{
source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh;
conda activate pth20_old;

export TOKENIZERS_PARALLELISM=false;
python stage2.py --devices '0,1,2,3' --filename "molt5_pubchem" --opt_model 'laituan245/molt5-large' --mode eval --prompt '[START_SMILES]{}[END_SMILES]. ' --tune_gnn --llm_tune full --inference_batch_size 2 --root "data/PubChemDataset_v4" --batch_size 8 --accumulate_grad_batches 4 --peft_config scripts/lora_config_t5.json --precision '16' --stage2_path "all_checkpoints/molt5_pubchem/last-v1.ckpt" --max_epochs 200 --caption_eval_epoch 100 --optimizer adafactor;


# python stage2.py --devices '0,1,2,3' --filename "molt5_pubchem_adafactor" --opt_model 'laituan245/molt5-large' --mode eval --prompt '[START_SMILES]{}[END_SMILES]. ' --tune_gnn --llm_tune full --inference_batch_size 2 --root "data/PubChemDataset_v4" --batch_size 8 --accumulate_grad_batches 4 --peft_config scripts/lora_config_t5.json --precision '16' --stage2_path "all_checkpoints/molt5_pubchem_adafactor/last.ckpt" --max_epochs 200 --caption_eval_epoch 100 --optimizer adafactor;



# python stage2.py --devices '0,1,2,3' --filename "molt5_chebi_adafactor" --opt_model 'laituan245/molt5-large' --mode ft --prompt '[START_SMILES]{}[END_SMILES]. ' --tune_gnn --llm_tune full --inference_batch_size 2 --root "data/ChEBI-20_data" --batch_size 8 --accumulate_grad_batches 4 --peft_config scripts/lora_config_t5.json --precision '16' --stage2_path "all_checkpoints/molt5_stage2/last.ckpt" --max_epochs 100 --caption_eval_epoch 100 --optimizer adafactor;

exit
}