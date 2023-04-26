# GraphTextPretrain

This repository contains the pretrain code for the paper "Natural Language-informed Understanding of Molecule Graphs”

# ****Requirements****

```python
Ubuntu 16.04.7
python 3.9.12
cuda 10.1

# pytorch
pip install torch==1.8.1+cu101 torchvision==0.9.1+cu101 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# torch_geometric 
# you can download the following *.whl files in https://data.pyg.org/whl/
pip install torch_cluster-1.5.9-cp39-cp39-linux_x86_64.whl
pip install torch_scatter-2.0.8-cp39-cp39-linux_x86_64.whl
pip install torch_sparse-0.6.12-cp39-cp39-linux_x86_64.whl
pip install torch-geometric

# transformers (4.18.0)
pip install transformers 

# rdkit
pip install rdkit-pypi

# ogb
pip install ogb

# pytorch_lightning (1.6.2)
pip install pytorch_lightning 
```

# Acknowledgment

This repository uses some code from [Graphormer](https://github.com/microsoft/Graphormer/). Thanks to the original authors for their work!

# Directory Structure

Although we only use the GIN with 5 layers and a 300-dimensional hidden size as the graph encoder, we also provide the Graphormer model and the GIN model with virtual nodes.

Prior to the pre-training procedure, we initialize the GIN model using the GraphCL checkpoint and the BERT model using the checkpoints of Sci-BERT or KV-PLM. You should download the GIN, SciBert, KVPLM checkpoint into the gin_pretrained/, bert_pretrained/, kvplm_pretrained/ folders, respectively. These checkpoints can be downloaded on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**.

```python
--GraphTextPretrain
  --data
  --graph # the folder contains the graph data
  --textx # the folder contains the text data
  --bert_pretrained # the folder contains SciBert checkpoint
  --kvplm_pretrained # the folder contains KVPLM checkpoint
  --gin_pretrained # the folder contains GIN checkpoint
  --data_provider # dataset and datamodule
    --pretrain_datamodule.py
    --pretrain_dataset.py
  --model
    --gin # the GIN model
	  --conv.py
	  --gnn.py
    --graphormer # the Graphormer model
	  --graphormer_graph_encoder.py
	  --graphormer_graph_encoder_layer.py
	  --graphormer_layers.py
	  --multihead_attention.py
    --bert.py
    --gin_model.py
    --contrastive_gin.pyx # contrastive learning
    --contrastive_gin_virturalnode.py
    --contrastive_graphformer.py
  --utils
    --GraphAug.py # the functions of graph augmentation
    --lr.py # control learning rate 
  --train_gin.py # the pretrain script
```

# ****Data****

Since the dataset files are too large, we just provide about one hundred graph-text pairs which can be used to test the pretrain process.

Please download the full data files on [the Baidu Netdisk](https://pan.baidu.com/s/1hYbVOMhr7pwLkUtl5FWqbg), the password is **1234**.

# Pretrain

To jointly pretrain the text encoder (Bert which is initized by the KV-PLM checkpoint) and the graph encoder (GIN) on eight gpus, run this command:

```
python train_gin.py --batch_size=32 --accelerator='gpu' --gpus='0,1,2,3,4,5,6,7' --graph_self --max_epochs=300 --num_workers=8
```

To jointly pretrain the text encoder (Bert which is initized by the SciBert checkpoint) and the graph encoder (GIN) on eight gpus, run this command:

```
python train_gin.py --batch_size=32 --accelerator='gpu' --gpus='0,1,2,3,4,5,6,7' --graph_self --max_epochs=300 --num_workers=8 --bert_pretrain
```

# Our Pretrained models

We provide two pretrained models using our method. You can download them on [the Baidu Netdisk](https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A), the password is **1234**. All our downstream tasks use these two models.

Pretrained model when Bert is initized by the **KV-PLM** checkpoint:

```
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

Pretrained model when Bert is initized by the **SciBert** checkpoint:

```
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

# Citation

```
@article{su2022molecular,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```