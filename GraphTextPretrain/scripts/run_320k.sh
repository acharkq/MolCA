{

python train_blip2.py --warmup_steps 1000 --root 'data/PubChemDataset/PubChem-320k' --max_epochs 50 --devices '3,4' --batch_size 64 --precision 32 --filename gtc
exit
}