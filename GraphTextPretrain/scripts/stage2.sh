{
filename='gal13_v2';

# pretrain on stage1 checkpoint
python pretrain_stage2.py --devices '6,7' --filename "pt_${filename}" --stage1_path "all_checkpoints/mola_dataset/epoch=49-step=120950.ckpt" --opt_model facebook/galactica-1.3b --max_epochs 10 --mode pretrain;

# fine-tune on the pretrain's checkpoint
python pretrain_stage2.py --devices '6,7' --filename "ft_${filename}" --init_checkpoint "all_checkpoints/pt_${filename}/last.ckpt" --opt_model facebook/galactica-1.3b --max_epochs 5 --mode ft --scheduler None;
exit
}
