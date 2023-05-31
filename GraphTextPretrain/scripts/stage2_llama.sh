{
filename='llama13b';
llm="./llms/vicuna-13b";
devices='[1,]';

# pretrain on stage1 checkpoint
python pretrain_stage2_vicuna.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/mola_dataset_notune_gnn_nosampline/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt 'The molecule\t' --tune_gnn --batch_size 2 --load_in_8bit;

# # fine-tune on the pretrain's checkpoint
# python pretrain_stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 10 --mode ft --tune_gnn;

# fine-tune on the pretrain's checkpoint
# python pretrain_stage2_vicuna.py --devices $devices --filename "ft_${filename}_lora_tuning" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --lora_tuning;
exit
}
