# Pytorch-Lightning-Face-Recognition
The Objective is to implement Siamese Network for Face Recognition using Pytorch Lightning.

# Siamese Network
Siamese Network is one of the simplest neural network architecture. It involves two identical Convolutional Neural Network which shares same weight as it gets trained. It takes in two inputs either of same class or of different class with respective label as 0 and 1. Since we are performing face recognition and also as suggested in paper, if two input images are of same person then the difference between them should be 0 and vice versa.

![Siamese Network Architecture](/images/siamese_network.png)

**In most cnn architecture, we predict class of the input images like dog or cat etc, but here the network outputs a vector for each input image, over which we calculate the pairwise distance between two vector and the pairwise distance is in turn passed to contrastive loss function for optimization.**

# Dataset

[Try out any of these datasets from Kaggle](https://www.kaggle.com/c/deepfake-detection-challenge/discussion/121594)

# How to execute
  > python model.py --batch_size=64 --pretrain_epochs=1000 --margin=2.0 --imageFolderTrain='./SiameseNetworkData/training/' --imageFolderTest='./SiameseNetworkData/testing/' --learning_rate=5e-5 --resize=100

# Result
![Face Recognition](/images/siamese_result.png)

