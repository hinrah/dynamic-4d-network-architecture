# 4D dynamic network architectures

 **This is a fork of [dynamic_network_architectures](https://github.com/MIC-DKFZ/dynamic_network_architectures)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends acvl_utils to support 4D convolutions. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.

 Large parts of this README are derived from the original dynamic_network_architectures documentation.


# Dynamic Network Architectures

This repository contains several ResNet, U-Net and VGG architectures in pytorch that can be dynamically adapted to a varying number of image dimensions (1D, 2D or 3D, 4D) and the number of input channels.

## Available models
### ResNet NOT 4D
We implement the standard [ResNetD](https://arxiv.org/pdf/1812.01187.pdf) 18, 34, 50 and 152. For ResNets 50 and 152 also bottleneck implementations are available. Moreover, adapted versions that are better suited for smaller image sizes such as CIFAR can be used.

All models additionally include regularization techniques like [Stochastic Depth](https://arxiv.org/pdf/1603.09382.pdf), [Squeeze & Excitation](https://arxiv.org/pdf/1709.01507.pdf) and [Final Layer Dropout](https://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf). 

### VGG NOT 4D
In contrast to the original [VGG](https://arxiv.org/pdf/1409.1556.pdf) implementation we exclude the final fully-connected layers in the end and replace it by additional convolutional layers and only one fully-connected layer in the end. Adapted versions that are better suited for smaller image sizes such as CIFAR can be used.

### U-Net
For the [U-Net](https://arxiv.org/pdf/1505.04597.pdf) a plain convolutional encoder as well as a residual encoder are available. 

# Acknowledgements

 **This is a fork of [dynamic_network_architectures](https://github.com/MIC-DKFZ/dynamic_network_architectures)**, originally
 developed by the Division of Medical Image Computing, German Cancer Research
 Center (DKFZ), and licensed under the Apache License 2.0.

 This fork extends acvl_utils to support 4D convolutions. See
 [CHANGES.md](CHANGES.md) for a list of modifications.

 This project is **not affiliated with, endorsed by, or maintained by** the
 original nnU-Net authors or the DKFZ. Please direct issues with this fork here,
 not to the upstream repository.