from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np

def masked_normalization(X, Y, scaler):

    # This function normalizes datasets without the including of 0's
    # Input of scaler as 'minmax' will result in MinMaxScaler
    # Input of scaler as 'ss' will result in StandardScaler
    X, Y = np.copy(X), np.copy(Y)
    nz = np.any(X, -1)

    if scaler == 'minmax':
        sc = MinMaxScaler().fit(X[nz])
    else:
        sc = StandardScaler().fit(X[nz])

    X[nz] = sc.transform(X[nz])
    Y[np.any(Y, -1)] = sc.transform(Y[np.any(Y, -1)])

    return X, Y
