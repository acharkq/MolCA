{
source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh;
conda activate pth20;

python main.py --test_dataset ../GraphTextPretrain/data/PubChemDataset/PubChem-320k/test --init_checkpoint ../GraphTextPretrain/all_checkpoints/tune_gnn320k
}