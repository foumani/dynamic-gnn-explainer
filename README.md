# Generating data-sets
"generator" module creates datasets.
```python
from generator.DynamicGraphGenerator import DynamicGraphGenerator
dyn_generator = DynamicGraphGenerator(T=5, size=200, n_motif=200, p_dynamic_noise=0.1)
```
In this case you have 5 snapshots with 200 length base graph that contains 200 double_house-dynamic_motif with 10% noise that is added at the end in the shape of random edges.
By running cells from "create_dataset.ipynb" you can save these datasets for the usage of evolveGCN.

# EvolveGCN
This repository contains our modified code of [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191), published in AAAI 2020.

## Data


## Requirements
  * PyTorch 1.0 or higher
  * Python 3.6

## Set up with Docker

This docker file describes a container that allows you to run the experiments on any Unix-based machine. GPU availability is recommended to train the models. Otherwise, set the use_cuda flag in parameters.yaml to false.

### Requirements

- [install docker](https://docs.docker.com/install/)
- [install nvidia drivers](https://www.nvidia.com/Download/index.aspx?lang=en-us)

### Installation

#### 1. Build the image

From this folder you can create the image

```sh
sudo docker build -t gcn_env:latest docker-set-up/
```

#### 2. Start the container

Start the container

```sh
sudo docker run -ti  --gpus all -v $(pwd):/evolveGCN  gcn_env:latest
```

This will start a bash session in the container.

## Usage

Set --config_file with a yaml configuration file to run the experiments. For example:

```sh
python run_exp.py --config_file ./evolvegcn/experiments/parameters_explanation_egcn_h.yaml
```

Most of the parameters in the yaml configuration file are self-explanatory. For hyperparameters tuning, it is possible to set a certain parameter to 'None' and then set a min and max value. Then, each run will pick a random value within the boundaries (for example: 'learning_rate', 'learning_rate_min' and 'learning_rate_max').

Setting 'use_logfile' to True in the configuration yaml will output a file, in the 'log' directory, containing information about the experiment and validation metrics for the various epochs. The file can be manually analyzed.



## Reference

[1] Aldo Pareja, Giacomo Domeniconi, Jie Chen, Tengfei Ma, Toyotaro Suzumura, Hiroki Kanezashi, Tim Kaler, Tao B. Schardl, and Charles E. Leiserson. [EvolveGCN: Evolving Graph Convolutional Networks for Dynamic Graphs](https://arxiv.org/abs/1902.10191). AAAI 2020.
