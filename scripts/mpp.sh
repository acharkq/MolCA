{

# python icl_mpp.py --filename mpp_test --init_checkpoint "./all_checkpoints/caption_default/epoch=99-step=18700.ckpt" --llm_tune lora --peft_dir all_checkpoints/caption_default/lora_epoch_99 --device 0 --mode opt

# python icl_mpp.py --filename mpp_test --init_checkpoint "./all_checkpoints/caption_default/epoch=99-step=18700.ckpt" --device 0 --mode opt

# python icl_mpp.py --filename mpp_test --init_checkpoint "all_checkpoints/ft_gal1.3b_smiles_prompt_freeze_long/last.ckpt" --device 0 --mode blip

python icl_mpp.py --filename mpp_test --device 0 --mode opt --opt_model 'facebook/galactica-6.7b' --batch_size 2;

# python mol_qa.py --init_checkpoint "./all_checkpoints/stage2_default/epoch=9-step=48380.ckpt"  --device 0

# python mol_qa.py --device 0
exit
}
