{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
# llm='facebook/galactica-125m';
devices='6,7';

# pretrain on stage1 checkpoint
# python stage2.py --devices $devices --filename "pt_${filename}_full" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt 'The molecule\t' --tune_gnn --llm_tune full --inference_batch_size 4;


# python stage2.py --devices $devices --filename "ft_${filename}_full" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --llm_tune full --inference_batch_size 4;


# python stage2.py --devices $devices --filename "pt_${filename}_lora" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt 'The molecule\t' --tune_gnn --llm_tune lora --inference_batch_size 8;


python stage2.py --devices $devices --filename "ft_${filename}_loralora" --stage2_path "all_checkpoints/pt_${filename}_lora/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --llm_tune lora --inference_batch_size 8 --peft_dir "all_checkpoints/pt_${filename}_lora/lora_epoch_9";

exit
}
