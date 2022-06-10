***
# Collaborative_Filtering
### Recommendation Systems based on Collaborative Filtering with Pytorch
- Train and Evaluate

***
## This repository contains

### - Trainer.py 
- overll training process as Class including</br>
  pre-processing, set dataloader, set model, training, evaluate, save result

### - Generator
- Custom Torch Iterable Dataset Class
- support Pointwise, Pairwise, Matrix

### - Models
- Custom Torch nn.Module class
- support SVD, SVD++, BPR, GMF, MLP, NeuMF, AutoRec, CDAE, NGCF

### - example
- jupyter notebook
- can follow overall training process

### - data
- include dataset
- can download at

   |Dataset|Download|
   |:---:|---:|
   |movie-lens 1M|[link](https://grouplens.org/datasets/movielens/)|

***
## Installation

The code was tested on Ubuntu 20.04 and Windows 10, with [Anaconda](https://www.anaconda.com/download) Python 3.7 and [PyTorch]((http://pytorch.org/)) v1.11.0.

After install Anaconda:

0. [Optional but recommended] Create a new conda environment. 

    ~~~
    conda create --name reco_sys python=3.7
    ~~~
    And activate the environment.
    
    ~~~
    conda activate reco_sys
    ~~~

1. Install requirements:

    ~~~
    pip install -r requirements.txt
    ~~~


## Getting started with jupyter-notebook
|Name|example with notebook|
|:---:|---|
|**SVD**|[examples/SVD.ipynb](examples/SVD.ipynb)|
|**SVD++**|[examples/SVDpp.ipynb](examples/SVDpp.ipynb)|
|**BPR**|[examples/BPR.ipynb](examples/BPR.ipynb)|
|**GMF**|[examples/GMF.ipynb](examples/GMF.ipynb)|
|**MLP**|[examples/MLP.ipynb](examples/MLP.ipynb)|
|**NeuMF**|[examples/NeuMF.ipynb](examples/NeuMF.ipynb)|
|**AutoRec**|[examples/AutoRec.ipynb](examples/AutoRec.ipynb)|
|**CDAE**|[examples/CDAE.ipynb](examples/CDAE.ipynb)|
|**NGCF**|[examples/NGCF.ipynb](examples/NGCF.ipynb)|


## Getting started with Trainer.py
1. Edit config.py for Trainer.py
   
2. Edit Models/config.py for algorithm

3. Put data file at below directory
   ~~~
   data/...
   ~~~

4. Run
   ~~~
   $python Trainer.py --data_path="data/movie_lens/ratings.csv" --user_colname="userId" --item_colname="movie_id" --rating_colname="rating"
   ~~~

***
