{
# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;
# cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;


filename='gal1.3b';
llm='facebook/galactica-1.3b';
devices='[7]';

# python ft_clf.py --devices $devices --filename "ft_fg_regression_${filename}" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 10 --task_type regression --root data/PubChemDataset_v4/;
# python ft_clf.py --devices $devices --filename "ft_fg_regression_${filename}" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 10 --task_type regression --root data/PubChemDataset_v4/;
# python ft_clf.py --devices $devices --filename "ft_fg_regression_${filename}" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 10 --task_type regression --root data/PubChemDataset_v4/;

# python ft_clf.py --devices $devices --filename "ft_fg_smiels_regression_${filename}"  --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 100 --task_type regression --root data/PubChemDataset_v4/ --smiles_only --seed 0 --peft_dir "all_checkpoints/smiles_pt_gal1.3b/lora_epoch_9";

# python ft_clf.py --devices $devices --filename "ft_fg_smiels_regression_${filename}" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 100 --task_type regression --root data/PubChemDataset_v4/ --smiles_only --seed 1 --peft_dir "all_checkpoints/smiles_pt_gal1.3b/lora_epoch_9";

python ft_clf.py --devices $devices --filename "ft_fg_smiels_regression_${filename}" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 100 --task_type regression --root data/PubChemDataset_v4/ --smiles_only --seed 2 --peft_dir "all_checkpoints/smiles_pt_gal1.3b/lora_epoch_9";

# python ft_clf.py --devices $devices --filename "ft_fg_regression_${filename}" --stage2_path "all_checkpoints/pt_${filename}_correct_tunegnn/last.ckpt" --opt_model $llm --mode ft --tune_gnn --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --init_lr 1e-4 --scheduler None --llm_tune lora --max_epochs 100 --task_type regression --root data/PubChemDataset_v4/ --seed 2;


exit
}
