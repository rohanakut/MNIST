# MNIST
The repo contains various algorithms which were initially applied on mnist dataset using tensorflow

The repository contains 3 algorithms namely Logistic Regression, Artificial neural networks and Convolutional neural networks.
It is observed that the Artificial neural networks perform much better on the test set compared to logistic regression.
The accuracy obtained using ANN is close to 95% while that obtained using logistic regression is 92%.

Even though the ANN is superior in terms of accuracy compared to logistic regression the time required to train the algorith is much more than that required by logistic regression.

In case of CNN I have used Batch Normalisation technique and dropouts have been introduced in order to prevent overfitting.I have used Keras over Tensorflow for training CNN.I have traind CNN for 5 epochs with a batch size of 64. 

## Time comparison:

Logistic regression : 4.98sec

Artificial Neural Nets(4 hidden layers) : 32.99sec

Convolutional Neural Networks( 4 convolutional layers and 1 Artificial neural net layer) : 500s(approx) per epoch

## Accuracy:

Logistic regression : 91.31%

Artificial Neural Nets(4 hidden layers) : 94.22% 

Convolutional Neural Network : 99.55%

## Installation Instruction:

The code has been written and executed using [anaconda](https://conda.io/docs/user-guide/install/windows.html) and spyder.Tensorflow has been installed externally in anaconda evnvironment using these [instructions](https://www.tensorflow.org/install/install_windows).I would suggest not installing tensorflow through pip and use native anaconda instead. 
