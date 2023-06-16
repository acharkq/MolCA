{

# python mol_qa.py --init_checkpoint "./all_checkpoints/caption_default/epoch=99-step=18700.ckpt" --llm_tune lora --peft_dir all_checkpoints/caption_default/lora_epoch_99 --device 0

python mol_knn_token.py --init_checkpoint "all_checkpoints/pt_gal1.3b_correct_tunegnn/last.ckpt"  --device 6 --prompt '[START_I_SMILES]{}[END_I_SMILES]. '

# python mol_knn_token.py --init_checkpoint "all_checkpoints/ft_chebi_gal1.3b_correct_tunegnn/last.ckpt"  --device 6 --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --llm_tune lora --peft_dir "all_checkpoints/ft_chebi_gal1.3b_correct_tunegnn/lora_epoch_99"

# python mol_knn_token.py --init_checkpoint "all_checkpoints/caption_default/last.ckpt"  --device 6 --prompt 'The molecule\t' --llm_tune lora --peft_dir "all_checkpoints/caption_default/lora_epoch_99"

exit
}
