# compressml
Image compression, upscaling using Machine learning. Models can be configured.

# Models
This library makes use of several models for different use cases.

## MLLAE

MLLAE is a (Mostly) LossLess AutoEncoder model. It is trained by a neural network that attempts to map an image X -> X. The compression is accomplished by the network architecture: there is a bottleneck layer L<sub>k</sub> where bytes(L<sub>k</sub>> < bytes(X). Compressing an image applies the first portion of the network architecture, with the output of L<sub>k</sub> being the compressed representation. Decompression requires applying the remaining part of the network.

### MLLAE-N

MLLAE-N lets you train a new MLLAE model that compresses images to a fixed size of at most N bytes. Training a new MLLAE-N model will attempt to optimize compression on the training dataset for that fixed size; note that there are theoretical limits on lossless compression in the general case, and that you may overfit to your training data.

## LAE

LAE is a Lossy AutoEncoder model. Its encryption works the same way as with MLLAE. It is trained differently: rather than attempting to minimize data loss for a fixed bottleneck layer size, it maximizes reward as a function g(L, N) where L is data loss and N is the size. This way, the model makes tradeoffs where loss is acceptable if it can be justified by a reduced compression size. The default LAE model uses well-tuned function g.

### LAE-g

LAE-g lets you train your own LAE model using your own definition of g. You can enforce maximum loss, minimum compression size, etc. by tuning G.

## RED

RED is a Resolution Enhancement Decoder. It takes low resolution images and attempts to resolve them to images of higher resolution. It is trained by lowering the resolution of images, and then creating a neural network to map images back to their original resolutions.
