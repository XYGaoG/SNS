## Introduction

This repository is the implementation of SNS. 


## Requirements

All experiments are implemented in Python 3.9 with Pytorch 1.11.0.

To install requirements:
```setup
pip install -r requirements.txt
```

## Data

The datasets used in our experiments are download from the repository of [SHGP (NeurIPS 2022)](https://arxiv.org/abs/2210.10462): [[github]](https://github.com/kepsail/SHGP). The label rate is fixed at 6\% for all datasets.

We provide the code of dataset processing in `./scr/load_data.py`.


## Model training

To train the model and execute the SNS:

```bash
$ cd ./scr
$ bash ./run.sh
```
