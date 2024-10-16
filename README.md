## Overview
This repository contains code for training a classification model using a custom dataset. The model is implemented using PyTorch, with data loading, training, validation, and testing procedures. This README will guide you through understanding the structure and functionality of the code.

## Dataset
The dataset used for this project is located in the Dataset/Data_for_UCI_named.csv file. This is a CSV file containing labeled data for a classification task. The dataset is split into training, validation, and test sets.

## Steps
0. Load Data
The DataSet class is used to load the dataset

1. Split Data
The dataset is split into three parts:

* Training set: 7000 samples
* Validation set: 1000 samples
* Test set: 2000 samples

2. DataLoader
The DataLoader is used to batch the data and shuffle it:

* Training Loader: Batches of 500 samples with shuffling.
* Validation Loader: A single batch of the full validation set without shuffling.
* Test Loader: A single batch of the full test set without shuffling.

3. Initialize Model
An instance of the Model class is created, which defines the neural network architecture.

4. Define Loss Function and Optimizer
The loss function used is CrossEntropyLoss, and the optimizer can be chosen between Adam and SGD. In this case, SGD with a learning rate of 0.01 is used.

5. Training Loop
The model is trained for a maximum of 500 epochs. During each epoch:

The optimizer performs gradient descent based on the computed loss.
The model is validated every 50 epochs, and if the validation accuracy exceeds 95%, the training stops early.

6. Test the Model
After training, the model is tested on the test set. The accuracy of the model on the test data is printed.

Instructions
* Clone this repository.
* Install the necessary Python packages:
```bash
pip install torch
```
Place your dataset in the Dataset folder, or change the file path in the code.
Run the script to start training and testing the model.
```bash
python Main.py
```
