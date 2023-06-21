{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='6,7';

python stage2.py --devices $devices --filename "nopretrain_lora_${filename}"  --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --inference_batch_size 8 --max_epochs 100;

exit
}
