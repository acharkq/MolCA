{
filename='gal13_notune_tune';
llm='facebook/galactica-1.3b';
devices='6,7';

# pretrain on stage1 checkpoint
python pretrain_stage2.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/mola_dataset_tune_gnn_nosampline/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain;

# # fine-tune on the pretrain's checkpoint
# python pretrain_stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 10 --mode ft --tune_gnn;

# fine-tune on the pretrain's checkpoint
python pretrain_stage2.py --devices $devices --filename "ft50_${filename}" --init_checkpoint "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 50 --mode ft --tune_gnn;
exit
}
