{

python train_blip2_stage2.py --devices '6,7' --filename stage2_test --num_query_token 8 --stage1_path all_checkpoints/full_nodelip_q8_f16/epoch=49-step=101450.ckpt --opt_model facebook/galactica-30b --batch_size 2
exit
}