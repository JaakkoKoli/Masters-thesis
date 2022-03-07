import numpy as np
import pandas as pd

def isPositiveDefinite(A):
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False

def gpr(X_test, theta, X, Y, k):
    Xstar = np.matrix(X_test)
    K = k(X, X, theta)
    if isPositiveDefinite(K):
        KK_inv = np.linalg.cholesky(K)
    else:
        KK_inv = np.linalg.pinv(K)
    #KK_inv = np.linalg.inv(K)
    Ky = KK_inv.dot(Y.T)
    mus = [0]*np.shape(Xstar)[0]
    sigs = [0]*np.shape(Xstar)[0]
    
    
    for i in range(np.shape(Xstar)[0]):
        XX = Xstar[i]
        Kstar = k(X, XX, theta)
        Kstarstar = k(XX,XX,theta)
        mu = Kstar.T.dot(Ky)
        cv = Kstarstar - (Kstar.T.dot(KK_inv).dot(Kstar))
        if cv < 0:
            cv = abs(cv)
        mus[i] = float(mu)
        sigs[i] = float(cv)
    return pd.DataFrame({"mu": mus, "sig": sigs})