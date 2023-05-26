{

# python train_blip2_stage2.py --devices '4,5' --filename gal13 --stage1_path all_checkpoints/full_nodelip_q8_warup1e6/epoch=49-step=101450.ckpt --opt_model facebook/galactica-1.3b --batch_size 32

# python train_blip2_stage2.py --devices '6,7' --filename gal67 --stage1_path all_checkpoints/full_nodelip_q8_warup1e6/epoch=49-step=101450.ckpt --opt_model facebook/galactica-6.7b --batch_size 32


python pretrain_stage2.py --devices '6,7' --filename gal30 --stage1_path all_checkpoints/full_nodelip_q8_warup1e6/epoch=49-step=101450.ckpt --opt_model facebook/galactica-30b --batch_size 8 --accumulate_grad_batches 4
exit
}
