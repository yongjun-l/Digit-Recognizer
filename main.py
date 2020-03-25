import numpy as np
import pandas as pd
import matplotlib as plt
import NN_model as nn


prediction = pd.read_csv("sample_submission.csv")
test_csv = pd.read_csv("test.csv")
train_csv = pd.read_csv("train.csv")

train = train_csv.values.T  # turn train set data frame to numpy array
test = test_csv.values.T
y_values = train[[0], :]  # bring y values  [3,1,4,6,2,0,...]
train = train[1:, :]

y = np.zeros((10, y_values.shape[1]))
for i in range(y_values.shape[1]):
    y[y_values[0][i]][i] = 1  # one-hot encoding

print('y : ', y.shape)
print('train : ', train.shape)
print('test : ', test.shape, type(test))


# scaling data set values to range (0,1)
train = np.divide(train, 255)
#test_norm = np.divide(test, 255)  # getting memory error on this line

# running
layer_dims = [784, 500, 300, 100, 50, 10]
parameters = nn.model(train, y, layer_dims)
data_to_submit = nn.predict(parameters, layer_dims, train, y)
correct = data_to_submit.where(data_to_submit.loc[:, 'prediction'] == train_csv.loc[:, 'label'])
correct_number = correct.count(numeric_only=True)
print("correct  : ", correct_number)
print("accuracy : ", 100 * (correct_number / y.shape[1]))
print('finished')
