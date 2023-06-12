{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
# llm='facebook/galactica-125m';
devices='6,7';

# python stage2.py --devices $devices --filename "ft_${filename}_iupac_slora_smiles_prompt" --stage2_path "all_checkpoints/stage2_default/last.ckpt" --opt_model $llm --max_epochs 100 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8 --iupac_prediction;


### evaluation
python stage2.py --devices $devices --filename "gal13b_zs_iupac_eval${filename}" --opt_model $llm --max_epochs 100 --stage2_path "all_checkpoints/stage2_default/last.ckpt" --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]## Chemical and Physical Properties. The following are chemical properties for' --iupac_prediction --inference_batch_size 16 --llm_tune freeze;

exit
}
