{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
# llm='facebook/galactica-125m';
devices='4,5';


python stage2.py --devices $devices --filename "pt_${filename}_nostage1" --opt_model $llm --max_epochs 10 --mode pretrain --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune freeze --inference_batch_size 8;

python stage2.py --devices $devices --filename "ft_${filename}_nostage1" --stage2_path "all_checkpoints/pt_${filename}_nostage1/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --inference_batch_size 8 --max_epochs 100;


python stage2.py --devices $devices --filename "ft_chebi_${filename}_nostage1" --stage2_path "all_checkpoints/pt_${filename}_nostage1/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --inference_batch_size 8 --max_epochs 100 --root "data/ChEBI-20_data" --peft_config "PeftConfig/CheBI.json" ;
exit
}
