{
filename='llama13b';
llm="./llms/vicuna-13b";
devices='6,7';

# pretrain on stage1 checkpoint
# python stage2_vicuna.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/mola_dataset_notune_gnn_nosampline/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode eval --prompt 'The molecule\t' --tune_gnn;

# python stage2_vicuna.py --devices $devices --filename "pt_${filename}" --stage2_path "all_checkpoints/pt_llama13b/last.ckpt" --opt_model $llm --max_epochs 10 --mode eval --prompt 'The molecule\t' --tune_gnn;


# # fine-tune on the pretrain's checkpoint
# python pretrain_stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 10 --mode ft --tune_gnn;

# fine-tune on the pretrain's checkpoint
python stage2_vicuna.py --devices $devices --filename "ft_${filename}_lora_tuning" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --lora_tuning --batch_size 8 --accumulate_grad_batches 4;
exit
}
