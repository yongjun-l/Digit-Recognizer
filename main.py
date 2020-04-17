"""
Written by Yongjun Lee

description - This is a implementation of MNIST digit data set classifier using softmax regression model.
"""

import numpy as np
import pandas as pd
import NN_model as nn

# reading the data training set as pandas.Dataframe
prediction = pd.read_csv("sample_submission.csv")
test_csv = pd.read_csv("test.csv")
data_csv = pd.read_csv("train.csv")

# turn the data's datatype from pd.Dataframe to np.array so it is easier to work with
data = data_csv.values.T
test = test_csv.values.T

# separating the data between pixel values, labels / training, development set
train = data[1:, 2000:]
dev = data[1:, :2000]
label_train_numeric = data[[0], 2000:]
label_dev_numeric = data[[0], :2000]


# create one-hot encoded version of the label in order to calculate loss with softmax
label_train_onehot = np.zeros((10, label_train_numeric.shape[1]))
for i in range(label_train_numeric.shape[1]):
    label_train_onehot[label_train_numeric[0][i]][i] = 1  # one-hot encoding


label_dev_onehot = np.zeros((10, label_dev_numeric.shape[1]))
for i in range(label_dev_numeric.shape[1]):
    label_train_onehot[label_dev_numeric[0][i]][i] = 1  # one-hot encoding


# check to see everything is in desired shape
print('train', train.shape)
print('label_train_numeric', label_train_numeric.shape)
print('label_train_onehot', label_train_onehot.shape)
print('dev', dev.shape)
print('label_dev_numeric', label_dev_numeric.shape)
print('label_dev_onehot', label_dev_onehot.shape)


# scaling pixel values to range [0,1]
train = np.divide(train, 255)
dev = np.divide(dev, 255)


# training the model, and making a prediction
layer_dims = [784, 500, 300, 100, 50, 10]
parameters = nn.model(train, label_train_onehot, layer_dims, num_iter=200, optimizer='momentum', show_cost=True,
                      learning_rate=0.1, minibatch_size=2048, beta1=0.9)
data_to_submit = nn.predict(parameters, layer_dims, dev, label_dev_onehot)

data_to_submit.to_csv('myscv.csv')
a = data_to_submit.loc[:, 'prediction']
b = data_csv.iloc[:2000, 0]
correct = data_to_submit.where(a == b)
correct_number = correct.count(numeric_only=True)
print("correct  : ", correct_number)
print("accuracy : ", 100 * (correct_number / label_dev_onehot.shape[1]))
