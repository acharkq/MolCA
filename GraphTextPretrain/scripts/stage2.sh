{

python train_blip2_stage2.py --devices '0,1,2,3' --filename stage2_test --num_query_token 4 --stage1_path all_checkpoints/tune_gnn320k/epoch=49-step=135250.ckpt
exit
}