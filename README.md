# tf-3dgan

[![license](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/meetshah1995/tf-3dgan/blob/master/LICENSE)
[![arXiv Tag](https://img.shields.io/badge/arXiv-1610.07584-brightgreen.svg)](https://arxiv.org/abs/1610.07584)

## Tensorflow implementation of 3D Generative Adversarial Network.

This is a tensorflow implementation of the paper "Learning a Probabilistic Latent Space of Object Shapes 
via 3D Generative-Adversarial Modeling"

[web](http://3dgan.csail.mit.edu/)  

![](http://3dgan.csail.mit.edu/images/model.jpg)


### Requirements

* tensorflow>=1.0
* visdom>=1.0.1 (for mesh visualization)
* scipy
* scikit-image
* stl (optional)


#### One-line installation
    ```
    pip install scipy scikit-image stl visdom
    ```

### Data

* Download the training data from the 3D Shapenet [website](http://3dshapenets.cs.princeton.edu/3DShapeNetsCode.zip)
* Extract the zip and modify the path appropriately in `dataIO.py`

### Usage

To train the model

```
python 3dgan_mit_biasfree.py 0 <path_to_model_checkpoint>
```

To generate chairs

```
python 3dgan_mit_biasfree.py 1 <path_to_trained_model>
```


### Source code files

| File      | Description                                                                   |
|-----------|-------------------------------------------------------------------------------|
|3dgan_mit_biasfree.py      | 3dgan as mentioned in the paper, with same hyperparams. 
|3dgan.py                   | baseline 3dgan with fully connected layer at end of discriminator.
|3dgan_mit.py               | 3dgan as mentioned in the paper with bias in convolutional layers.
|3dgan_autoencoder.py       | 3dgan with support for autoencoder based pre-training.
|3dgan_feature_matching.py  | 3dgan with additional loss of feature mathcing of last layers. 
|dataIO.py                  | data input output and plotting utilities.
|utils.py                   | tensorflow utils like leaky_relu and batch_norm layer.


### Todo

* Host the trained models
* Add argparser based interface
* Add threaded dataloader
* Release the pytorch and keras versions of the GAN.
