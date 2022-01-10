import numpy as np

gridSize_1 = 8 #base 1
gridSize_0 = 7 #base 0
uniqueOptions = gridSize_1**2

def rbf(x1,x2,theta):
    #if np.shape(x1)[1] != np.shape(x2)[1]:
    #    print("incorrect x1 and x2 input dimensions")
    #    return
    #elif any([x<0 for x in theta]):
    #    print("all parameters must be greater than 0")
    #    return
    
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
		
