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
		

def shepard(x1, x2, theta):
    N1 = np.shape(x1)[0]
    N2 = np.shape(x2)[0]
    d = np.shape(x1)[1]
    sigma = np.zeros((N1, N2))
    sf = theta[d]
	p_minowski = theta[d+2]
    for i in range(d):
        sigma = sigma + ((abs(np.matrix((np.subtract.outer(x1[:,i],x2[:,i]))))**p_minowski)**(1/p_minowski)**p_minowski)/theta[i].T
    if np.array_equal(x1,x2):
        return sf*np.exp(-0.5 * sigma) + theta[d+1]*np.identity(N1)
    else:
        return sf*np.exp(-0.5 * sigma)
		
def oru(x1,x2,theta):
    N1 = np.shape(x1)[0]
    N2 = np.shape(x2)[0]
    d = np.shape(x1)[1]
    sigma = np.zeros((N1, N2))
    sf = theta[d]
    for i in range(d):
        sigma = sigma + abs(np.matrix(np.subtract.outer(x1[:,i],x2[:,i])/theta[i])).T
    if np.array_equal(x1,x2):
        return sf*np.exp(-0.5 * sigma) + theta[d+1]*np.identity(N1)
    else:
        return sf*np.exp(-0.5 * sigma)
		
def matern(x1,x2,theta):
    N1 = np.shape(x1)[0]
    N2 = np.shape(x2)[0]
    d = np.shape(x1)[1]
    sigma = np.zeros((N1, N2))
    sf = theta[d]
    for i in range(d):
        sigma = sigma + abs(np.matrix(np.subtract.outer(x1[:,i],x2[:,i])/theta[i])).T
    if np.array_equal(x1,x2):
        return sf*(1+(np.sqrt(3)*sigma))*exp(-np.sqrt(3)*sigma) + theta[d+1]*np.identity(N1)
    else:
        return sf*(1+(np.sqrt(3)*sigma))*exp(-np.sqrt(3)*sigma)