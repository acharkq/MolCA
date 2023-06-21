{
filename='molt5large';
llm='laituan245/molt5-large';

devices='4,5';

# pretrain on stage1 checkpoint
# python llm_tuning.py --devices $devices --filename "smiles_pt_${filename}" --llm_name $llm --max_epochs 10 --mode pretrain --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora;


python llm_tuning.py --devices $devices --filename "smiles_molt5_ft${filename}" --llm_name $llm --max_epochs 100 --mode ft --prompt '{}' --llm_tune full --inference_batch_size 8 --init_lr 1e-4 --scheduler None;

exit
}
