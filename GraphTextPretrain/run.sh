# source /home/users/nus/e0517239/scratch/anaconda3/etc/profile.d/conda.sh;
source /home/users/nus/e0517239/.bashrc;
conda activate pth20;
cd /home/project/11002701/zyliu/Mol-BLIP2/GraphTextPretrain;

python run.py;

nvidia-smi;

gpustat;