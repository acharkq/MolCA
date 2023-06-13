# Text-to-graph molecule generation
The PyTorch implementation of MoMu and Moflow-based zero-shot text-to-graph molecule generation, described in "Natural Language-informed Understanding of Molecule Graphs".



## License & disclaimer
The codes can be used for research purposes only. This package is strictly for non-commercial academic use only.



## Acknowledgments
We adapted the code of the PyTorch implementation of MoFlow which is publicly available at https://github.com/calvin-zcx/moflow. Please also check the license and usage there if you want to make use of this code. 



## Install
* Operating system: Linux version 4.18.0-80.7.1.el8_0.x86_64, with a single NVIDIA Titan RTX GPU, cuda 11.2; also tested on Linux version 4.15.0-189-generic, using a single NVIDIA TITAN V GPU, CUDA Version 10.1.243.
* Please refer to https://github.com/calvin-zcx/moflow for the requirements. We use the same packages as follows:
```
conda create --name TGgeneration python pandas matplotlib  (conda 4.6.7, python 3.8.5, pandas 1.1.2, matplotlib  3.3.2)
conda activate TGgeneration
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch (pytorch 1.6.0, torchvision 0.7.0)
conda install rdkit  (rdkit 2020.03.6)
conda install orderedset  (orderset 2.0.3)
conda install tabulate  (tabulate 0.8.7)
conda install networkx  (networkx 2.5)
conda install scipy  (scipy 1.5.0)
conda install seaborn  (seaborn 0.11.0)
pip install cairosvg (cairosvg 2.4.2)
pip install tqdm  (tqdm 4.50.0)
```

* Our implementation also requires the following additional dependencies or packages (torch-geometric, transformers, spacy):
```
pip install torch_scatter-2.0.6-cp38-cp38-linux_x86_64.whl
pip install torch_sparse-0.6.9-cp38-cp38-linux_x86_64.whl
pip install torch_cluster-1.5.9-cp38-cp38-linux_x86_64.whl
pip install torch_spline_conv-1.2.1-cp38-cp38-linux_x86_64.whl
pip install torch-geometric
pip install transformers
pip install spacy
```
(The .whl files can be downloaded from https://pytorch-geometric.com/whl/torch-1.6.0%2Bcu101.html 
For other cuda versions, please select from https://pytorch-geometric.com/whl/)

It takes about half an hour to install all the packages.


## Prepare pre-trained models
#### Downloading the MoFlow model trained on the zinc250k dataset in 
```
https://drive.google.com/drive/folders/1runxQnF3K_VzzJeWQZUH8VRazAGjZFNF 
``` 
Put the folder "zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask" in the folder ./MoleculeGeneration/results 

#### Downloading the pre-trained graph and text encoders of MoMu
Put the pretrained files "littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt" for MoMu-S and "littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt" in the folder ./MoleculeGeneration
(Download from https://pan.baidu.com/s/1jvMP_ysQGTMd_2sTLUD45A  password: 1234)
Pretrained model when Bert is initized by the **KV-PLM** checkpoint:

```python
checkpoints/littlegin=graphclinit_bert=kvplm_epoch=299-step=18300.ckpt
```

Pretrained model when Bert is initized by the **SciBert** checkpoint:

```python
checkpoints/littlegin=graphclinit_bert=scibert_epoch=299-step=18300.ckpt
```

#### Downloading the per-trained Bert model
Download the folder "bert_pretrained" from https://huggingface.co/allenai/scibert_scivocab_uncased 
Put the folder "bert_pretrained" in the folder ./MoleculeGeneration



## Testing & Useage 
#### Generating molecules with the query texts used in the paper:
default: MoMu-S; To use MoMu-K, uncomment line 683 and comment line 682 in Graph_generate.py
```
cd MoleculeGeneration
python Graph_generate.py --model_dir results/zinc250k_512t2cnn_256gnn_512-64lin_10flow_19fold_convlu2_38af-1-1mask  -snapshot model_snapshot_epoch_200 --gpu 0  --data_name zinc250k --hyperparams-path moflow-params.json   --temperature 0.85  --batch-size 1 --n_experiments 5  --save_fig true --correct_validity true
```

#### Generating molecules with the query texts
Put the custom text descriptions in the list in line 816-825 of Graph_generate.py.


#### Results
The generated 60 (the number of generated molecules can be specified in lines 834-835 of Graph_generate.py) molecule graphs with respect to the {id}-the text description are saved in the subfolder "generated/sci/text_{id}/" of the folder "MoleculeGeneration". 
The corresponding SMILES and negative similarities between the text and the molecule graph are also output. For example, for the 0-th input text description, the output has the following forms:
```
0
['O[IH]CI(O)CC[IH]OI=CF', ... , 'CC(CCCOO)O[IH]OI=[IH](C)CC[IH]O']
[-2.299729347229004, ... , -2.235506772994995]
```
It takes about half an hour to generate 60 molecule graphs given an input text description.



## Citation
Please cite the following paper if you use the codes:

```
@article{su2022natural,
  title={Natural Language-informed Understanding of Molecule Graphs},
  author={Bing Su, Dazhao Du, Zhao Yang, Yujie Zhou, Jiangmeng Li, Anyi Rao, Hao Sun, Zhiwu Lu, Ji-Rong Wen},
  year={2022}
}
```