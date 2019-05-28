# compressml
Image compression and resolution enhancement using Machine learning. Create a compression model specific to your data with configurable models.

# Models
This library makes use of several models for different use cases. Detailed descriptions and explanations can be found in resources/docs/ at a later date.

## MLLAE

MLLAE is a (Mostly) LossLess AutoEncoder model. It is trained by a neural network that attempts to map an image X -> X. The compression is accomplished by the network architecture: there is a bottleneck layer L<sub>k</sub> where bytes(L<sub>k</sub>) < bytes(X). Compressing an image applies the first portion of the network architecture (the encoder), with the output of L<sub>k</sub> being the compressed representation. Decompression requires applying the remaining part of the network, the decoder.

### MLLAE-N

MLLAE-N lets you train a new MLLAE model that compresses images to a fixed size of at most N bytes. Training a new MLLAE-N model will attempt to optimize compression on the training dataset for that fixed size; note that there are theoretical limits on lossless compression in the general case, and that you will probably overfit to your training data. This is essentially a more restricted case of LAE-g.

## LAE

LAE is a Lossy AutoEncoder model. Its encryption works the same way as with MLLAE. It is trained differently: rather than attempting to minimize data loss for a fixed bottleneck layer size, it maximizes reward as a function g(L, N) where L is data loss and N is the size. This way, the model makes tradeoffs where loss is acceptable if it can be justified by a reduced compression size. The default LAE model will use a well-tuned function g.

### LAE-g

LAE-g lets you train your own LAE model using your own definition of the model loss function g. You can enforce maximum loss, minimum compression size, etc. by tuning g.

## RED

RED is a Resolution Enhancement Decoder. It takes low resolution images and attempts to resolve them to images of higher resolution. It is trained by lowering the resolution of images, and then creating a neural network to map images back to their original resolutions.

# APIs

A long term goal of this project is to provide simple apis in a variety of programming languages to use pretrained or custom trained models. 

# Setup
The long term goal is to enable python environment setup by running a single script, and to have premade models able to be downloaded and used using well-known package managers. Currently, this is not implemented. 

## Guide
Please download and install Anaconda for Python 3.X (https://www.anaconda.com/distribution/). Then, run `conda install keras` and `conda install tensorflow`. Jupyter is included in Anaconda by default, and with properly installed anaconda, can be started with `jupyter notebook` from the terminal; this is used to open .ipynb files. There will probably be other third party modules that in various files; unless otherwise noted, these will be available through conda or pip. Please feel free to contribute set up scripts that make using, training, or reconfiguring these models simpler.

## Operating Systems

Initially, the only supported operating systems will be commonly used linux distributions (Ubuntu) and macOS. If you have a fix to restore functionality for a specific operating system (preferably by using more generic code), please submit a PR. If something is not working on macOS or a linux distribution such as Ubuntu, please raise an issue.
