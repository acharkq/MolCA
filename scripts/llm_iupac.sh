{
filename='gal1.3b';
llm='facebook/galactica-1.3b';

devices='4,5';

# pretrain on stage1 checkpoint
# python llm_tuning.py --devices $devices --filename "smiles_pt_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora;


# python llm_tuning.py --devices $devices --filename "smiles_ft_iupac_slora${filename}" --llm_name $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --iupac_prediction --inference_batch_size 8;

### evaluation
python llm_tuning.py --devices $devices --filename "smiles_zs_eval${filename}" --llm_name $llm --max_epochs 100 --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]## Chemical and Physical Properties. The following are chemical properties for' --iupac_prediction --inference_batch_size 16 --llm_tune full;

exit
}
