import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore")

# from subprocess import check_output
# print(check_output(["ls", ["../"]]).decode('utf8'))

X_l = np.load('X.npy')
y_l = np.load('Y.npy')

# Join a sequence of arrays along an row axis.
X = np.concatenate((X_l[204:409], X_l[822:1027] ), axis=0) # from 0 to 204 is zero sign and from 205 to 410 is one sign 
z = np.zeros(205)
o = np.ones(205)
Y = np.concatenate((z, o), axis=0).reshape(205,1)    # .reshape(X.shape[0],1)
print("X shape: " , X.shape)
print("Y shape: " , Y.shape)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=42)

number_of_train = X_train.shape[0]
number_of_test = X_test.shape[0]

X_train_flatten = X_train.reshape(number_of_train, X_train.shape[1] * X_train.shape[2])
X_test_flatten = X_test.reshape(number_of_test, X_test.shape[1] * X_test.shape[2])

print("X train flatten",X_train_flatten.shape)
print("X test flatten",X_test_flatten.shape)

x_train = X_train_flatten.T
x_test = X_test_flatten.T
y_train = y_train.T
y_test = y_test.T








def initialize_parameters_and_layer_sizes_NN(x_train, y_train):
    parameters : {
        "weight1":np.random.randn(3, x_train.shape[0]) * 0.1,
        "bias1":np.zeros(3, 1),
        "weight2":np.random.randn(y_train.shape[0], 3) * 0.1,
        "bias2":np.zeros(y_train.shape[0], 1)
    }
    return parameters

# print(np.random.randn(3, 4096).shape) # (3, 4096) 3 Rows and 4096 Columns

# w1 = 3 X 4096 | b1 = 3 X 1 | A = 3 X 1 | w2 = 1 X 3 | b2 = 1 X 1  
# Output = Probability

def sigmoid(z):
    return 1 / 1 + (np.exp(-z))


def forward_propagation_NN(x_train, parameters):
    z1 = np.dot(parameters["weight1"], x_train)
    A1 = np.tanh(z1)

    z2 = np.dot(parameters["weight2"], A1)
    A2 = sigmoid(z2)

    cache = {
        z1 : z1, # (3 X 1)
        A1 : A1, # (3 X 1)
        z2 : z2, # (1)
        A2 : A2  # (1)
    }
    return A2, cache


# Compute cost
def compute_cost_NN(A2, Y, parameters):
    logprobs = np.multiply(np.log(A2),Y)
    cost = -np.sum(logprobs)/Y.shape[1]
    return cost

# print(np.sum(np.array([0.3, 0.4, 0.7]),keepdims=True)) | [1.4]

# print((1 - np.power([3, 4, 5], 2))) | [ -8 -15 -24]

def backward_propagation_NN(parameters, cache, X, Y):

    dZ2 = cache["A2"] - Y
    dW2 = np.dot(cache["A2"].T, dZ2) / X.shape[1]
    db2 = np.sum(dZ2, axis=1, keepdims=True) / X.shape[1]

    dZ1 = np.dot(parameters["weight2"].T, dZ2) * (1 - np.power(cache["A1"], 2))
    dW1 = np.dot(dZ1, X.T) / X.shape[1]
    db1 = np.sum(dZ1, axis=1, keepdims=True)

    grads = {
        "dweight1": dW1,
        "dbias1": db1,
        "dweight2": dW2,
        "dbias2": db2
    }
    return grads

def update_parameters_NN(parameters, grads, learning_rate = 0.01):

    parameters = {
        "weight1" : parameters["weight1"] - learning_rate * grads["dweight1"],
        "bias1" : parameters["bias1"] - learning_rate * grads["dbias1"],
        "weight2" : parameters["weight2"] - learning_rate * grads["dweight2"],
        "bias2" : parameters["bias2"] - learning_rate * grads["dbias2"]
    }
    return parameters

def predict_NN(parameters,x_test):
    A2, cache = forward_propagation_NN(x_test, parameters)

    y_prediction = np.zeros((1, x_test.shape[1]))
    for i in range(A2.shape[1]):
        if A2[0, i] <= 0.5:
            y_prediction[0, i] = 0
        else:
            y_prediction[0, i] = 0

    return y_prediction



def two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations):
    cost_list = []
    index_list = []
    #initialize parameters and layer sizes
    parameters = initialize_parameters_and_layer_sizes_NN(x_train, y_train)

    for i in range(0, num_iterations):
         # forward propagation
        A2, cache = forward_propagation_NN(x_train,parameters)
        # compute cost
        cost = compute_cost_NN(A2, y_train, parameters)
         # backward propagation
        grads = backward_propagation_NN(parameters, cache, x_train, y_train)
         # update parameters
        parameters = update_parameters_NN(parameters, grads)
        
        if i % 100 == 0:
            cost_list.append(cost)
            index_list.append(i)
            print ("Cost after iteration %i: %f" %(i, cost))
    plt.plot(index_list,cost_list)
    plt.xticks(index_list,rotation='vertical')
    plt.xlabel("Number of Iterarion")
    plt.ylabel("Cost")
    plt.show()
    
    # predict
    y_prediction_test = predict_NN(parameters,x_test)
    y_prediction_train = predict_NN(parameters,x_train)

    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_train - y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(y_prediction_test - y_test)) * 100))
    return parameters

parameters = two_layer_neural_network(x_train, y_train,x_test,y_test, num_iterations=2500)



