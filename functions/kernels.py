import numpy as np

def rbf(x1,x2,theta):
    N1 = np.shape(x1)[0]
    N2 = np.shape(x2)[0]
    d = np.shape(x1)[1]
    sigma = np.zeros((N1, N2))
    sf = theta[d]
    for i in range(d):
        sigma = sigma + np.matrix((np.subtract.outer(x1[:,i],x2[:,i])/theta[i])**2).T
    if np.array_equal(x1,x2):
        return sf*np.exp(-0.5 * sigma) + theta[d+1]*np.identity(N1)
    else:
        return sf*np.exp(-0.5 * sigma)