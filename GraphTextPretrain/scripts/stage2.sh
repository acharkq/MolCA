{

python train_blip2_stage2.py --devices '4,5' --filename gal13 --stage1_path all_checkpoints/full_nodelip_q8_warup1e6/epoch=49-step=101450.ckpt --opt_model facebook/galactica-1.3b --batch_size 32
exit
}