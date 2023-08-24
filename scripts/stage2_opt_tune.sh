{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='2,3';


# python stage2.py --devices $devices --filename "ft_mlora_${filename}_correct_tunegnn" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --inference_batch_size 8 --max_epochs 100 --peft_config ./PeftConfig/CheBIMiddle.json;



# python stage2.py --devices $devices --filename "test" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --inference_batch_size 8 --max_epochs 100 --batch_size 16  --lora_r 8 --peft_config "PeftConfig/CheBIMiddle.json"


## full tuning
python stage2.py --devices $devices --filename "ft_full_pubchem324k_${filename}" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune full --inference_batch_size 8 --max_epochs 100 --save_every_n_epochs 0


# python llm_tuning.py --devices $devices --filename "test" --llm_name $llm --max_epochs 100 --mode ft --prompt 'The SMILES of this molecule is [START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --batch_size 32 --lora_r 16 # --peft_config "PeftConfig/CheBIMiddle.json"

# --peft_dir "all_checkpoints/smiles_pt_${filename}/lora_epoch_9";
exit
}

# --peft_config ./PeftConfig/CheBIMiddle.json