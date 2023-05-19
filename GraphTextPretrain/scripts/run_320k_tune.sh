{
# source /gpu-work/gp9000/gp0900/next/anaconda3/etc/profile.d/conda.sh;
# conda activate pth20;

python train_320k.py --warmup_steps 1000 --root 'data/PubChemDataset/PubChem-320k' --max_epochs 50 --tune_gnn --filename tune_gnn320k
exit
}