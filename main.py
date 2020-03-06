import pandas as pd
import numpy as np
from math import *
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# read csv file
obs_data = pd.read_csv("./data/obs.csv")
array = obs_data.transpose().index.values
# timestamp
deltaT = 0.1
# define matrices
A = [[1, deltaT], [0, 1]]
B = [[0, 0], [0, deltaT]]


def gaussian(mean, sigma2, x):
    coefficient = 1 / (sqrt(2.0 * pi * sigma2))
    exponential = exp(-(x - mean)**2 / (2 * sigma2))
    return coefficient * exponential


def prediction2d(x, v, t, a):
    A = np.array([[1, t],
                  [0, 1]])
    X = np.array([[x],
                  [v]])
    U = np.array([[0], [a]])
    B = np.array([[0, 0], [0, deltaT]])
    X_prime = np.dot(A, X) + np.dot(B, U) + \
        multivariate_normal(mean=[0, 0], cov=np.array(
            [[0.057, 0.001], [0.001, 0.030]])).pdf([x, v])
    return X_prime


def covariance2d(sigma1, sigma2):
    cov1_2 = sigma1 * sigma2
    cov2_1 = sigma2 * sigma1
    cov_matrix = np.array([[sigma1 ** 2, cov1_2],
                           [cov2_1, sigma2 ** 2]])
    return cov_matrix


# initial state
mean = np.array([0, 0])
P = np.array([[4, 0], [0, 1]])
var = multivariate_normal(mean=mean, cov=P)
p = 0
v = 0
Q = np.array([[0.057, 0.001], [0.001, 0.03]])
X = np.array([[p], [v]])
R = np.array([[0.0025, 0.0025], [0.0025, 0.0025]])
for i in range(100):
    print(i)
    # Predicted state estimate
    # covariance problem: it shpuld be positive semi-definite --> I changed the first element to 0.057 isntead of 0.0
    X = prediction2d(X[0][0], X[1][0], deltaT, 0.1)
    # Predicted error covariance
    P = np.dot(A, P).dot(np.transpose(A)) + Q
    H = np.array([[1, 0], [0, 0]])
    S = H.dot(P).dot(H.T) + R
    # Kalman gain
    K = P.dot(np.transpose(H)).dot(np.linalg.inv(S))

    if(i % 10 == 0 and i != 0):

        # Reshape the new data into the measurement space.
        Y = float(array[i]) + gaussian(0, 0.0025, X[0][0])

        # Update the State Matrix
        # Combination of the predicted state, measured values, covariance matrix and Kalman Gain
        X = X + K.dot(Y - H.dot(X))

    # Updated error covariance
    P = (np.identity(len(K)) - K.dot(H)).dot(P)

    print("X : {}".format(X))
    print("P : {}".format(P))
