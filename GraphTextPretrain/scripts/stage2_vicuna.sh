{
source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
conda activate pth20;
cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;

filename='vicuna7b';
llm="./llms/vicuna-7b";
devices='0,1';

# pretrain on stage1 checkpoint
python stage2.py --devices $devices --filename "pt_${filename}" --stage1_path "all_checkpoints/stage1_default/epoch=49-step=120950.ckpt" --opt_model $llm --max_epochs 10 --mode pretrain --prompt 'The molecule\t' --tune_gnn --batch_size 16 --accumulate_grad_batches 2;


# fine-tune on the pretrain's checkpoint
# python stage2_vicuna.py --devices $devices --filename "ft_${filename}_lora_tuning" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --lora_tuning --batch_size 8 --accumulate_grad_batches 4;
for epoch in 09 19 29 39 49 59 69 79 89 99
do

# fine-tune on the pretrain's checkpoint without lora
python stage2.py --devices $devices --filename "ft_${filename}" --stage2_path "all_checkpoints/pt_${filename}/last.ckpt" --opt_model $llm --max_epochs 100 --mode ft --tune_gnn --prompt 'The molecule\t' --init_lr 1e-4 --scheduler None --batch_size 16 --accumulate_grad_batches 2;
exit
}
