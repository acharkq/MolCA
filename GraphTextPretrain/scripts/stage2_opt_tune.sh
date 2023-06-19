{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='4,5';

# pretrain on stage1 checkpoint
# python stage2.py --devices $devices --filename "pt_${filename}_correct_tunegnn" --stage1_path "all_checkpoints/stage1_default_tune_gnn/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune freeze --inference_batch_size 8;

python stage2.py --devices $devices --filename "ft_mlora_${filename}_correct_tunegnn" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --inference_batch_size 8 --max_epochs 100 --peft_config ./PeftConfig/CheBIMiddle.json;


# python stage2.py --devices $devices --filename "ft_chebi_mlora_${filename}_correct_tunegnn_scheduled" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --inference_batch_size 8 --max_epochs 100 --root "data/ChEBI-20_data" --peft_config ./PeftConfig/CheBIMiddle.json;
exit
}
