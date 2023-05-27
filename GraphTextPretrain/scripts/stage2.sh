{
filename='test';

# pretrain on stage1 checkpoint
python pretrain_stage2.py --devices '2,3' --filename "pt_${filename}" --init_checkpoint "all_checkpoints/gal13/epoch=9-step=40580.ckpt" --opt_model facebook/galactica-1.3b --batch_size 8 --accumulate_grad_batches 4 --max_epochs 10 --mode pretrain;

# fine-tune on the pretrain's checkpoint
python pretrain_stage2.py --devices '2,3' --filename "ft_${filename}" --init_model "all_checkpoints/pt_${filename}/last.ckpt" --opt_model facebook/galactica-1.3b --batch_size 8 --accumulate_grad_batches 4 --max_epochs 5 --mode ft --scheduler None;
exit
}
