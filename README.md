# tiramisu-keras-script

Original jupyter-notebook implementation is provided with this link.
[https://github.com/TOSHISTATS/Semantic-segmentation-by-Fully-Convolutional-DenseNet](https://github.com/TOSHISTATS/Semantic-segmentation-by-Fully-Convolutional-DenseNet)

tiramisu-keras-script is a slightly modified version of the above implementation.

## Setup

```
# clone camvid dataset from github
$ git clone https://github.com/mostafaizz/camvid.git
$ python setup_dataset.py
```

## Training

```
$ python train.py --batchsize 10 --epochs 1
```

## Prediction

```
$ python predict.py <model_name.hdf5>
```

