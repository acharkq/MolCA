{
# source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;

# python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/gtc_wo_declip_q4 --device 2

# python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/gtc_wo_declip --device 2;

# python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/gtc_bn --device 2;

python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/gtc --device 2;

python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/cl_gtm_lm_50k --device 2;
}