{
source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh;
conda activate pth20_old;
export TOKENIZERS_PARALLELISM=false;
filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='0,1,2,3';

# pretrain on stage1 checkpoint
# python llm_tuning.py --devices $devices --filename "smiles_pt_full_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full;

python llm_tuning.py --devices $devices --filename "smiles_ft_lora_chebi_${filename}" --llm_name $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --peft_dir "all_checkpoints/smiles_pt_gal1.3b/lora_epoch_9" --root "data/ChEBI-20_data" --precision '16'  --batch_size 8 --accumulate_grad_batches 4 --caption_eval_epoch 100;

# python llm_tuning.py --devices $devices --filename "smiles_ft_midlora_chebi_${filename}" --llm_name $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --root "data/ChEBI-20_data" --precision '16'  --batch_size 8 --accumulate_grad_batches 4 --peft_config "scripts/lora_config.json" --caption_eval_epoch 100;


# python llm_tuning.py --devices $devices --filename "smiles_ft_full_${filename}" --init_checkpoint "all_checkpoints/smiles_pt_full_${filename}/last.ckpt" --llm_name $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full --save_every_n_epochs 0;

exit
}
