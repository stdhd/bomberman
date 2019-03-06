
import numpy as np
import matplotlib.pyplot as plt

import scipy.sparse
import scipy.linalg
import scipy.sparse.linalg


# From Homework 7 in Fundamentals of Machine Learning WS 18/19

# Kernel function K:
# Either x1 or x2 (exclusively) can be Nx2 arrays of values to compute
# multiple values of K(x) at the same time

def K(x1, x2, sig):
    sq_dist = np.sum((x1 - x2)**2, axis=-1, dtype=float)
    # Cutoff:
    sq_dist[sq_dist > 30 * sig**2] = np.inf

    return np.exp(-0.5 * sq_dist / sig**2)

# Return a scipy sparse matrix, containting the pairwise kernelized distances

def K_sparse_pairwise(X_new, X_training, sig):

    """
    Computes matrix of pairwise kernelized distances.
    :param X_new: New observation+action (one element only)
    :param X_training: Dataset of observations+actions
    :param sig: Kernel function hyperparameter
    :return: K(new, x_i) row vector, x_i in training set
    """
    N1, N2 = X_new.shape[0], X_training.shape[0]
    result = scipy.sparse.dok_matrix((N1,N2))

    indxs1, indxs2 = np.arange(N1), np.arange(N2)

    # For each row of the matrix, populate the entries that are != 0
    for i in indxs1:
        k = K(X_new[i], X_training, sig)
        result[i, indxs2[k!=0]] = k[k!=0]

    return result.tocsc()


# Weight vector (alpha on the exercise sheet)
def weight_vect(X, y, sig, tau):
    N = X.shape[0]
    indxs = np.arange(N)

    # Sparse kernel matrix G
    G = K_sparse_pairwise(X, X, sig)

    # Add tau to the diagonal
    G[indxs, indxs] += tau

    # ~100x faster than explicitly inverting
    return scipy.sparse.linalg.spsolve(G, y)
