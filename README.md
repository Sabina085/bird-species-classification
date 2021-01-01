# bird-species-classification
### Fine-grained classification on a subset of the Caltech-UCSD Birds-200-2011 dataset using transfer learning ### 

This repository contains the implementation of the method which achieves the second highest score (place 2/143, 91.436% accuracy) on the Kaggle competition www.kaggle.com/c/mva-recvis-2019/, as a part of the coursework for the Object Recognition and Computer Vision class, Master MVA, ENS Paris-Saclay. The challenge consists in designing a model that achieves the highest accuracy on a fine-grained classification task, using a subset of the Caltech-UCSD Birds-200-2011 bird dataset [[1]](#1).

An ensemble of 5 convolutional neural networks (CNNs) pre-trained on the dataset iNat2017 (three Inception-V3 CNNs and two Inception-V4 CNNs models, pre-trained using different input sizes) [[2]](#2) (https://github.com/richardaecn/cvpr18-inaturalist-transfer) is first used to extract features. The extracted features are then concatenated and fed into a simple multi-layer perceptron, with one hidden layer.  

## References
<a id="1">[1]</a> 
Wah C., Branson S., Welinder P., Perona P., Belongie S., “The Caltech-UCSD Birds-200-2011 Dataset.”, Computation & Neural Systems Technical Report, CNS-TR-2011-001

<a id="2">[2]</a> 
Yin Cui, Yang Song, Chen Sun, Andrew Howard, Serge Belongie. "Large scale fine-grained categorization and domain-specific transfer learning", CVPR, 2018.
