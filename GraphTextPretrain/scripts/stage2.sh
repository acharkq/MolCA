{
filename='stage1_notunegnn_v2';
llm='facebook/galactica-1.3b';
devices='0,1';

# pretrain on stage1 checkpoint
# python pretrain_stage2.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/mola_dataset_notune_gnn_nosampline/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --tune_gnn --mode pretrain --prompt 'The molecule\t';

# # fine-tune on the pretrain's checkpoint
# python pretrain_stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 10 --mode ft --tune_gnn;

# fine-tune on the pretrain's checkpoint
python pretrain_stage2.py --devices $devices --filename "ft_${filename}_lora_tuning"  --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --lora_tuning --batch_size 2
exit
}
