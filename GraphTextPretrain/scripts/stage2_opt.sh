{
filename='gal30b';
# llm='facebook/galactica-6.7b';
llm='facebook/galactica-30b';
devices='4,5,6,7';

# pretrain on stage1 checkpoint
# python stage2.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt 'The molecule\t' --batch_size 8 --accumulate_grad_batches 4;

# evaluate
python stage2.py --devices $devices --filename "pt_${filename}" --stage2_path "all_checkpoints/pt_gal30b/last.ckpt" --opt_model $llm --max_epochs 10 --mode eval --prompt 'The molecule\t' --batch_size 8 --accumulate_grad_batches 4 --inference_batch_size 1;

# # fine-tune on the pretrain's checkpoint
# python pretrain_stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 10 --mode ft --tune_gnn;

# fine-tune on the pretrain's checkpoint
# python stage2.py --devices $devices --filename "ft_${filename}_full_moreeval" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 20 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --llm_tune full --caption_eval_epoch 1 --inference_batch_size 32;

python stage2.py --devices $devices --filename "ft_${filename}_lora" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 50 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --llm_tune lora --batch_size 4 --accumulate_grad_batches 8 --inference_batch_size 1 --caption_eval_epoch 50;

exit
}
