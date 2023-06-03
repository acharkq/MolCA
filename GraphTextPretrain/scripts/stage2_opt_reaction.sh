{
filename='gal13_reaction_v2';
llm='facebook/galactica-1.3b';
devices='4,5';

# pretrain on stage1 checkpoint
# python stage2_reaction.py --devices $devices --filename "stage2_${filename}" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 2 --mode pretrain --prompt 'The molecule\t' --tune_gnn --reaction_weight 0.1;

# python stage2_reaction.py --devices $devices --filename "stage2_${filename}" --stage1_path "all_checkpoints/stage2_gal13_reaction/last.ckpt" --opt_model $llm --max_epochs 10 --mode eval --prompt 'The molecule\t' --tune_gnn;
# fine-tune on the pretrain's checkpoint

python stage2.py --devices $devices --filename "caption_${filename}_lora_tuning" --stage2_path "all_checkpoints/stage2_${filename}/epoch=99-step=18700.ckpt" --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --llm_tune lora;


exit
}
