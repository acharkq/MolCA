{
filename='molt5small';
llm='laituan245/molt5-small';

devices='6,7';

# pretrain on stage1 checkpoint
# python llm_tuning.py --devices $devices --filename "smiles_pt_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora;


python llm_tuning.py --devices $devices --filename "smiles_molt5_ft_iupac${filename}" --llm_name $llm --max_epochs 100 --mode ft --prompt 'The SMILES of the molecule is {}. ' --llm_tune full --iupac_prediction --inference_batch_size 8;

exit
}
