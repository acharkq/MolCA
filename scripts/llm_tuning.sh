{
filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='4,5';

# pretrain on stage1 checkpoint
python llm_tuning.py --devices $devices --filename "smiles_pt_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora;


python llm_tuning.py --devices $devices --filename "smiles_ft_${filename}" --init_checkpoint "all_checkpoints/smiles_pt_${filename}/last.ckpt" --llm_name $llm --max_epochs 100 --mode ft --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --peft_dir "all_checkpoints/smiles_pt_${filename}/lora_epoch_9";

exit
}
