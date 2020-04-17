import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def sigmoid(matrix):
    s = 1 / (1 + np.exp(-matrix))
    return s


def relu(matrix):
    matrix = matrix * (matrix > 0)
    return matrix


def softmax(matrix):
    exp_matrix = np.exp(matrix)

    softmax_matrix = exp_matrix / np.sum(exp_matrix, axis=0, keepdims=True)
    return softmax_matrix


def zero_init(dim):
    zero_vector = np.zeros((dim, 1))
    return zero_vector


# weight initialization method from He normal initialization (He-et-al)
def he_init(layer_dims):
    parameters = {}
    L = len(layer_dims)
    for i in range(1, L):
        parameters['w' + str(i)] = np.random.randn(layer_dims[i], layer_dims[i - 1]) * np.sqrt(1 / layer_dims[i - 1])
        parameters['b' + str(i)] = zero_init(layer_dims[i])
    return parameters


def init_momentum(layer_dims):
    momentum = {}
    L = len(layer_dims)
    for i in range(1, L):
        momentum['dw' + str(i)] = np.zeros((layer_dims[i], layer_dims[i - 1]))
        momentum['db' + str(i)] = np.zeros((layer_dims[i], 1))
    return momentum


# w : weights b : bias a : activation from previous layer
def forward_propagation(w, b, a, activation):
    z = np.dot(w, a) + b

    if activation == 'sigmoid':
        a = sigmoid(z)
    elif activation == 'relu':
        a = relu(z)
    elif activation == 'softmax':
        a = softmax(z)
    return z, a


def forward_propagation_full(parameters, layer_dim, x):
    cache = {'a' + str(0): x}  # stores z,a values for backward propagation
    L = len(layer_dim)
    for i in range(1, L):
        w = parameters['w' + str(i)]
        b = parameters['b' + str(i)]
        if i == L - 1:  # use softmax activation for the final layer
            cache['z' + str(i)], cache['a' + str(i)] = forward_propagation(w, b, cache['a' + str(i - 1)], 'softmax')
        else:  # use relu for the other layers.
            cache['z' + str(i)], cache['a' + str(i)] = forward_propagation(w, b, cache['a' + str(i - 1)], 'relu')

    aL = cache['a' + str(L - 1)]  # aL : final activation. returned value of softmax
    return aL, cache


# cost function for binary classification. not used in this model
"""
def cost_function(aL, y):  # aL : activation from final layer, y : labels
    cost = np.zeros((aL.shape[0], 1))
    for i in range(y.shape[0]):
        cost[i][0] = -(np.dot(y[i][:], np.log(aL[i][:]).T) + np.dot((1 - y[i][:]), np.log(1 - aL[i][:]).T)) / y.shape[1]
    cost = np.linalg.norm(cost, ord=2)
    return cost
"""


def softmax_cost(aL, y):
    loss = np.sum(-y * np.log(aL), axis=0, keepdims=True)
    cost = (1 / y.shape[1]) * np.sum(loss, axis=1)
    return cost


def backward_propagation(da, a_prev, z, w, b, activation_method):  # da : gradient of a from the next layer
    if activation_method == 'sigmoid':
        s = sigmoid(z)
        dz = da * s * (1 - s)

    elif activation_method == 'relu':
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0


    elif activation_method == 'softmax':
        s = softmax(z)
        dz = s - da

    m = a_prev.shape[1]
    dw = np.dot(dz, a_prev.T)
    da_prev = np.dot(w.T, dz)
    db = np.sum(dz, axis=1, keepdims=True) / m
    return da_prev, dw, db


def backward_propagation_full(x, y, parameters, cache):
    grads = {}
    L = int(len(parameters) / 2)

    grads['da' + str(L)] = y

    flag = False  # apply gradient descent for softmax only for the first case
    for i in reversed(range(1, L + 1)):
        if not flag:
            activation_method = 'softmax'
            flag = True
        else:
            activation_method = 'relu'

        da_prev, dw, db = backward_propagation(grads['da' + str(i)], cache['a' + str(i - 1)], cache['z' + str(i)],
                                               parameters['w' + str(i)], parameters['b' + str(i)], activation_method)
        grads['dw' + str(i)] = dw
        grads['db' + str(i)] = db
        grads['da' + str(i - 1)] = da_prev

        # print('backprop checking')
        # print('dw'+str(i), grads['dw'+str(i)].shape)
        # print('db'+str(i), grads['db'+str(i)].shape)
        # print('da'+str(i), grads['da'+str(i)].shape)

    return grads


def update_parameter(parameters, grads, learning_rate, layer_dims):
    L = len(layer_dims)
    for i in range(1, L):
        parameters['w' + str(i)] -= learning_rate * grads['dw' + str(i)]
        parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]
    return parameters

def update_momentum(momentum, grads, beta1, layer_dims):
    L = len(layer_dims)
    for i in range(1, L):
        momentum['dw' + str(i)] = (beta1 * momentum['dw' + str(i)]) + ((1 - beta1) * grads['dw' + str(i)])
        momentum['db' + str(i)] = (beta1 * momentum['db' + str(i)]) + ((1 - beta1) * grads['db' + str(i)])
    return momentum


def init_minibatch(x, y, minibatch_size):
    minibatches = []
    np.random.shuffle(x.T)
    np.random.shuffle(y.T)
    num_mini = y.shape[1] // minibatch_size
    for i in range(num_mini):
        mini_x = x[:, i * minibatch_size:(i + 1) * minibatch_size]
        mini_y = y[:, i * minibatch_size:(i + 1) * minibatch_size]
        minibatches.append((mini_x, mini_y))

    if y.shape[1] % num_mini != 0:
        mini_x = x[:, num_mini * minibatch_size:]
        mini_y = y[:, num_mini * minibatch_size:]
        minibatches.append((mini_x, mini_y))

    return minibatches


def model(x, y, layer_dims, num_iter=1, optimizer='momentum', show_cost=True, learning_rate=0.1, minibatch_size=2048,
          beta1=0.9):
    parameters = he_init(layer_dims)
    momentum = init_momentum(layer_dims)
    costs = []

    counter = 0
    for iter in range(num_iter):
        # create random minibatches
        minibatches = init_minibatch(x, y, minibatch_size)
        for minibatch in minibatches:
            mini_x, mini_y = minibatch
            aL, cache = forward_propagation_full(parameters, layer_dims, mini_x)

            # compute the cost and the gradient
            cost = softmax_cost(aL, mini_y)
            grads = backward_propagation_full(mini_x, mini_y, parameters, cache)

            # updating parameters
            if optimizer == 'momentum':
                momentum = update_momentum(momentum, grads, beta1, layer_dims)
                parameters = update_parameter(parameters, momentum, learning_rate, layer_dims)

            elif optimizer == 'gradient_descent':
                parameters = update_parameter(parameters, grads, learning_rate, layer_dims)

            # show the cost every 100 iterations
            if counter % 100 == 0 and show_cost == True:
                costs.append(cost)
                print('cost after {} iteration :'.format(counter), cost)

            counter += 1

    # plot how my cost is changing
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()

    return parameters


def predict(parameters, layer_dim, train, y):
    activation, cache = forward_propagation_full(parameters, layer_dim, train)
    prediction = activation.argmax(0)
    data_to_submit = pd.DataFrame(prediction, columns=['prediction'])
    return data_to_submit
