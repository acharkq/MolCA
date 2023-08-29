{

filename='gal1.3b';
# llm='facebook/galactica-1.3b';
llm='facebook/galactica-125M';
devices='2,3';

python stage2.py --devices '2,3' --filename "stage2_test" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full --inference_batch_size 8 --max_epochs 100 --save_every_n_epochs 0 --peft_config scripts/lora_config.json #--batch_size 16 --accumulate_grad_batches 2

}