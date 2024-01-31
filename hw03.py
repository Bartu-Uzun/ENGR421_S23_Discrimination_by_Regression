import math
import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg as linalg
import pandas as pd



X = np.genfromtxt("hw03_data_points.csv", delimiter = ",")
y = np.genfromtxt("hw03_class_labels.csv", delimiter = ",").astype(int)



i1 = np.hstack((np.reshape(X[np.where(y == 1)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 2)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 3)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 4)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 5)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 6)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 7)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 8)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 9)[0][0:5], :], (28 * 5, 28)),
                np.reshape(X[np.where(y == 10)[0][0:5], :], (28 * 5, 28))))

fig = plt.figure(figsize = (10, 5))
plt.axis("off")
plt.imshow(1 - i1, cmap = "gray")
plt.show()
fig.savefig("hw03_images.pdf", bbox_inches = "tight")



# STEP 3
# first 60000 data points should be included to train
# remaining 10000 data points should be included to test
# should return X_train, y_train, X_test, and y_test
def train_test_split(X, y):
    # your implementation starts below
    #print(X.shape)
    #print(y.shape)
    X_train = X[: 60000]
    y_train = y[: 60000]
    X_test = X[60000 :]
    y_test = y[60000 :]
    # your implementation ends above
    return(X_train, y_train, X_test, y_test)

X_train, y_train, X_test, y_test = train_test_split(X, y)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)



# STEP 4
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def sigmoid(X, W, w0):
    # your implementation starts below

    N = X.shape[0]
    K = W.shape[1]
    #X : N x D  W: D x K  w0: 1 x K

    # np.vstack((W, w0)): stacks W and w0 vertically --> we get the full W with each col corresponds to wc
    # np.hstack(X, np.ones((N, 1))): stacks X with a column of ones, so that when mult by the full W, we will get w*x+w0
    #scores = np.hstack((X, np.ones((N, 1)))) @ np.vstack((W, w0))

   

    scores = (1 / (1 + np.exp(-(np.matmul(X, W) + w0))))
    
    # your implementation ends above
    return(scores)



# STEP 5
# assuming that there are N data points and K classes
# should return a numpy array with shape (N, K)
def one_hot_encoding(y):
    # your implementation starts below
    N = y.shape[0]
    K = np.max(y)

    Y = np.zeros((N, K)).astype(int)

    Y[range(N), y - 1] = 1

    # your implementation ends above

    return(Y)



np.random.seed(421)
D = X_train.shape[1]
K = np.max(y_train)
Y_train = one_hot_encoding(y_train)
W_initial = np.random.uniform(low = -0.01, high = 0.01, size = (D, K))
w0_initial = np.random.uniform(low = -0.01, high = 0.01, size = (1, K))



# STEP 6
# assuming that there are D features and K classes
# should return a numpy array with shape (D, K)
def gradient_W(X, Y_truth, Y_predicted):
    # your implementation starts below


    A = (Y_truth - Y_predicted) * Y_predicted
    B = A * (1 - Y_predicted)
    
    gradient = - X.T @ B

    #print(gradient.shape)
    # your implementation ends above
    return(gradient)

# assuming that there are K classes
# should return a numpy array with shape (1, K)
def gradient_w0(Y_truth, Y_predicted):
    # your implementation starts below

    Ydiff = Y_truth - Y_predicted
    
    A = Ydiff * Y_predicted
    
    B = 1 - Y_predicted
   
    gradient = A * B
   
    gradient = -np.sum(gradient, axis=0)

    # your implementation ends above
    return(gradient)



# STEP 7
# assuming that there are N data points and K classes
# should return three numpy arrays with shapes (D, K), (1, K), and (200,)
def discrimination_by_regression(X_train, Y_train,
                                 W_initial, w0_initial):
    eta = 1.0 / X_train.shape[0]
    iteration_count = 200

    W = W_initial
    w0 = w0_initial

    # your implementation starts below

  
    objective_values = []
    while True:
        Y_predicted = sigmoid(X_train, W, w0)

        #print("y_predicted: ", Y_predicted[: 5])
        #print("iter count: ", iteration_count)

        #print("val to add: ", 0.5 * np.sum(np.square(Y_train - Y_predicted)))
        objective_values = np.append(objective_values, 0.5 * np.sum(np.square(Y_train - Y_predicted)))
        #print("objective_values: ", objective_values)
        


        W = W - eta * gradient_W(X_train, Y_train, Y_predicted)

        #print("W: ", W[10])
        w0 = w0 - eta * gradient_w0(Y_train, Y_predicted)

        #print("w0: ", w0)
        

        if iteration_count <= 1:
            
            break

        iteration_count -= 1

    
    # your implementation ends above
    return(W, w0, objective_values)

W, w0, objective_values = discrimination_by_regression(X_train, Y_train,
                                                       W_initial, w0_initial)
print(W)
print(w0)
print(objective_values[0:10])



fig = plt.figure(figsize = (10, 6))
plt.plot(range(1, len(objective_values) + 1), objective_values, "k-")
plt.xlabel("Iteration")
plt.ylabel("Error")
plt.show()
fig.savefig("hw03_iterations.pdf", bbox_inches = "tight")



# STEP 8
# assuming that there are N data points
# should return a numpy array with shape (N,)
def calculate_predicted_class_labels(X, W, w0):
    # your implementation starts below


    Y_predicted = sigmoid(X,W,w0)
    y_predicted = np.argmax(Y_predicted, axis = 1) + 1
    # your implementation ends above
    return(y_predicted)

y_hat_train = calculate_predicted_class_labels(X_train, W, w0)
print(y_hat_train)

y_hat_test = calculate_predicted_class_labels(X_test, W, w0)
print(y_hat_test)



# STEP 9
# assuming that there are K classes
# should return a numpy array with shape (K, K)
def calculate_confusion_matrix(y_truth, y_predicted):
    # your implementation starts below


    confusion_matrix = pd.crosstab(y_predicted, y_truth).to_numpy()
    # your implementation ends above
    return(confusion_matrix)

confusion_train = calculate_confusion_matrix(y_train, y_hat_train)
print(confusion_train)

confusion_test = calculate_confusion_matrix(y_test, y_hat_test)
print(confusion_test)
