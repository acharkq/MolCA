{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
# llm='facebook/galactica-6.7b';
llm='facebook/galactica-1.3b';
devices='0,1,2,3';

# pretrain on stage1 checkpoint
python stage2.py --devices $devices --filename "pt_${filename}_chebi_full_smiles_prompt" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 4 --root "data/ChEBI-20_data" --peft_config "PeftConfig/CheBI.json" ;


# python stage2.py --devices $devices --filename "pt_${filename}_lora" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES].' --tune_gnn --llm_tune lora --inference_batch_size 4;


# python stage2.py --devices $devices --filename "ft_${filename}_lora" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES].' --init_lr 1e-4 --scheduler None --llm_tune lora --inference_batch_size 4 --peft_dir "all_checkpoints/pt_${filename}/lora_epoch_9";

exit
}
