{
filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='6,7';

# pretrain on stage1 checkpoint
python llm_tuning.py --devices $devices --filename "smiles_pt_full_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full;


python llm_tuning.py --devices $devices --filename "smiles_ft_full_${filename}" --init_checkpoint "all_checkpoints/smiles_pt_full_${filename}/last.ckpt" --llm_name $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full --save_every_n_epochs 0;

exit
}
