# MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter


## Requirements

See `environment.yml`

## Dataset

Unzip the `./dataset.zip` under the `./data/` directory. It contains the molecule caption dataset of CheBI-20 and the retrieval dataset of PCDes and MoMu.

## Checkpoints

We share the checkpoints for reproducing results of molecule-text retrieval and for reproducing results of molecule captioning on the CheBI-20 dataset.

Please downlaod the checkpoint from this [link](https://ufile.io/6vffm5bg) and unzip it under the all_checkpoints directory


## Reproduce the results

1. Unzip the `./gin_checkpoint.zip` under the `./` directory.

2. Download the [Sci-BERT checkpoint](https://s3-us-west-2.amazonaws.com/ai2-s2-research/scibert/huggingface_pytorch/scibert_scivocab_uncased.tar) and unzip it under the `./bert_pretrained/` directory

### Molecule-Text Retrieval

Run the following script for evaluation on the PCDes dataset.

```python
python stage1.py --root 'data/kv_data' --gtm --lm --devices '0,1'  --filename pcdes_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval
```

Run the following script for evaluation on the MoMu dataset.

```python
python stage1.py --root 'data/kv_data' --gtm --lm --devices '0,1'  --filename pcdes_evaluation --init_checkpoint "all_checkpoints/share/stage1.ckpt" --rerank_cand_num 128 --num_query_token 8 --match_batch_size 64 --mode eval --use_phy_eval
```

### Molecule Captioning

Run the following script for evaluation on the CheBI-20 dataset.

```python
python stage2.py --devices '[0]' --filename "chebi_evaluation" --stage2_path "all_checkpoints/share/chebi.ckpt" --opt_model 'facebook/galactica-1.3b' --mode eval --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --inference_batch_size 8 --root "data/ChEBI-20_data" --peft_dir "all_checkpoints/share/chebi_lora";
```