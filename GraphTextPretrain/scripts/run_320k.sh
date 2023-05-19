{

python train_blip2.py --warmup_steps 1000 --root 'data/PubChemDataset/PubChem-320k' --max_epochs 50 --devices '4,5' --batch_size 64 --precision 32 --filename full_nodelip_q4  --gtm --lm --num_query_token 4;

python train_blip2.py --warmup_steps 1000 --root 'data/PubChemDataset/PubChem-320k' --max_epochs 50 --devices '4,5' --batch_size 64 --precision 32 --filename full_nodelip_q8  --gtm --lm --num_query_token 8;
exit
}