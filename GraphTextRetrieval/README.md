
# GraphTextRetrieval
Source code for cross-modality retrieval for *[Natural Language-informed Understanding of Molecule Graphs](https://arxiv.org/abs/2209.05481)*. 
Please go to [MoMu](https://github.com/ddz16/MoMu) to see the whole codebase.
## Workspace Prepare
If you want to explore our job, you can following the instructions in this section
- Step 1: Download the zip or clone the repository to your workspace.
- Step 2: Download the `littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt` and `littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt` from [BaiduNetdisk](https://pan.baidu.com/share/init?surl=jvMP_ysQGTMd_2sTLUD45A)(the Password is 1234). Create a new directory by `mkdir all_checkpoints` and then put the downloaded model under the directory. Rename `littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt` to `MoMu-K.ckpt` and `littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt` to `MoMu-S.ckpt`
- Step 3: Download files from [Sci-Bert](https://huggingface.co/allenai/scibert_scivocab_uncased/tree/main). Create a new directory by `mkdir bert_pretrained` and then put these files under the directory.
- Step 4: Install python environment. Some important requirements are listed as follows(In fact, the environment is the almost same as [GraphTextPretrain](https://github.com/ddz16/GraphTextPretrain), so you do not need to install again if you have follow its instructions):
  ```
  Ubuntu 16.04.7
  python 3.8.13
  cuda 10.1

  # pytorch
  pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

  # torch_geometric 
  # you can download the following *.whl files in https://data.pyg.org/whl/
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
  wget https://data.pyg.org/whl/torch-1.8.0%2Bcu101/torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
  pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
  pip install torch_scatter-2.0.8-cp38-cp38-linux_x86_64.whl
  pip install torch_sparse-0.6.12-cp38-cp38-linux_x86_64.whl
  pip install torch-geometric

  # transformers (4.18.0)
  pip install transformers 

  # rdkit
  pip install rdkit-pypi

  # ogb
  pip install ogb

  # pytorch_lightning (1.6.2)
  pip install pytorch_lightning 
## File Usage
The users may be going to use or edit the files below:
- main.py: Fine-tuning and testing code for cross-modality retrival. 
- data/
  - kv_data/: Pairs of (Graph, Text) data from  [KV-PLM](https://github.com/thunlp/KV-PLM) a.k.a PCdes
  - phy_data/: Pairs of (Graph, Text) data collected by us
- all_checkpoints/
  - MoMu-S.ckpt: Pretrained model  of MoMu-S
  - MoMu-K.ckpt: Pretrained model of MoMu-K
- data_provider/
  - match_dataset.py: Dataloader file
- model/
  - bert.py: Text encoder
  - gin_model.py: Graph encoder
  - constrastiv_gin.py Constrastive model with text encoder and graph encoder

## Zeroshot Testing
Zeroshot testing means cross-modality retrieval with origin MoMu. You can conduct zeroshot testing with differen settings as follows:
#### 1. zeroshot testing on phy_data with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
```
#### 2. zeroshot testing on phy_data with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
```
#### 3. zeroshot testing on kv_data with paragraph-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/kv_data/test
```
#### 4. zeroshot testing on kv_data with sentence-level:
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 1 --if_test 2 --if_zeroshot 1 --pth_test data/kv_data/test
```
## Finetuning and Testing
To make MoMu satisfy the cross-modality retrieval task better, you can finetune MoMu and then test. Befor fintuning, you should create a new directory to save finetuned model by `mkdir finetune_save`. 
#### 1. finetuning on kv_data with paragraph-level and testing:
```
# finetune MoMu and save as 'finetune_save/finetune_para.pt '
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_para.pt --data_type 0 --if_test 0 --if_zeroshot 0 

# test with fintuned model
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_para.pt --data_type 0 --if_test 2 --if_zeroshot 0
```
#### 2. finetuning on kv_data with sentence-level and testing:
```
# finetune MoMu and save as 'finetune_save/finetune_sent.pt '
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_sent.pt --data_type 1 --if_test 0 --if_zeroshot 0

# test with fintuned model
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --output finetune_save/finetune_sent.pt --data_type 1 --if_test 2 --if_zeroshot 0 
```
## Sample Result
Taking zeroshot testing on phy_data with paragraph-level as an example, we show the excuting result here.
It takes almost 10s to calculate the accuracy of retrieval, while calculating the Rec@20 takes about 2mins. 
```
python main.py --init_checkpoint all_checkpoints/MoMu-S.ckpt --data_type 0 --if_test 2 --if_zeroshot 1 --pth_test data/phy_data
Namespace(batch_size=64, data_type=0, epoch=30, graph_aug='dnodes', if_test=2, if_zeroshot=1, init_checkpoint='all_checkpoints/MoMu-S.ckpt', lr=5e-05, margin=0.2, output='finetune_save/sent_MoMu-S_73.pt', pth_dev='data/kv_data/dev', pth_test='data/phy_data', pth_train='data/kv_data/train', seed=73, text_max_len=128, total_steps=5000, warmup=0.2, weight_decay=0)
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 87/87 [00:16<00:00,  5.31it/s]
Test Acc1: 0.4565587918015103
Test Acc2: 0.4317727436174038
Rec@20 1: 0.4579036317871269
Rec@20 2: 0.4348471772743617
```
## Acknowledgment
This repository uses some code from [KV-PLM](https://github.com/thunlp/KV-PLM). Thanks to the original authors for their work!
## Citation
Please cite the following paper if you use the codes:

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```
