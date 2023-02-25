## Introduction
This repository is the implementation of SNS. 


## Requirements
All experiments are implemented in Python 3.9 with Pytorch 1.11.0.

To install requirements:
```setup
pip install -r requirements.txt
```

## Data

The datasets used in our experiments are download from the repository of NeurIPS 2022 paper: "Self-supervised Heterogeneous Graph Pre-training based on Structural Clustering": [[paper]](https://arxiv.org/abs/2210.10462) [[code]](https://github.com/kepsail/SHGP). The label rate is fixed at 6\% for all datasets.

We provide the code of dataset processing in 
```
./scr/utils/load_data.py
```
and also the processed labels when $imbalance \ ratio=0.1$ in 
```
./dataset
```


## Model training

To train the base model and execute the SNS:

```bash
$ cd ./scr
$ bash ./run.sh
```

## Acknowledgement

We would like to express our gratitude to the authors of following repositories for the open source code and datasets:

* [NeurIPS 22] "Self-supervised Heterogeneous Graph Pre-training based on Structural Clustering": [[paper]](https://arxiv.org/abs/2210.10462) [[code]](https://github.com/kepsail/SHGP).

* [TKDE 21] "Interpretable and Efficient Heterogeneous Graph Convolutional Network.": [[paper]](https://ieeexplore.ieee.org/document/9508875) [[code]](https://github.com/kepsail/ie-HGCN/). 

* [KDD 20] "Scaling Graph Neural Networks with Approximate PageRank.": [[paper]](https://arxiv.org/abs/2007.01570) [[code]](https://github.com/TUM-DAML/pprgo_pytorch). 
