import numpy as np
from functions.dataProcessing import *
from functions.kernels import *
from functions.models import *
from functions.acquisitionFunctions import *
import scipy.optimize as sco
import cma

def model(params, subjD, rounds, context, n=26, n2=4):
    # sample starting values from normal distribution
    tau, beta = np.exp(params)
    Xnew = [[x, y] for y in range(8) for x in range(8)]
    
    # Initialise lists for parameters and other data
    n_rounds = len(rounds)
    n_trials = n_rounds * 20 
    nll = [0]*n_rounds*19
    parameters = [0]*n_rounds*19

    
    for r in rounds:
        # Get data for round r
        roundD = (np.array(subjD["round"]) == r) & (np.array(subjD["context"]) == context)
        chosen = np.array(subjD["chosen"])[roundD]
        y = np.array(subjD["z"])[roundD]
        x1 = np.array(subjD["x"])[roundD]
        x2 = np.array(subjD["y"])[roundD]
        X = np.column_stack((x1,x2))
        
        # loop through trials 2-20 
        # (1 is considered to be a random choice since there is no information revealed yet)
        for i in range(19):
            # index for the round, needed if, for example, we want to look at rounds 2, 5, 9
            round_index = [x for x in range(len(rounds)) if rounds[x]==r][0]
            
            # Get values for beta and tau using lambda 
            X1 = np.matrix(X[0:i+1,:])
            y1 = np.matrix(y[0:i+1])
            
            nLL_lambda = [0]*n
            vals = np.linspace(0,4,n)
            vals = vals + vals[1]
            for ind, val in enumerate(vals):
                parVec = [val, val, 1, 0.0001]
                out = gpr(Xnew, parVec, X1, y1, rbf)

                utilityVec = ucb(out, [beta])
                utilityVec = utilityVec - np.max(utilityVec)

                p = np.exp(utilityVec / tau).tolist()[0]
                p = [max(0.00001,x) for x in p]
                p = p / np.sum(p)

                nLL_lambda[ind] = -np.log(p[chosen[i+1]])
            min_ind = int([x for x in range(n) if nLL_lambda[x]==min(nLL_lambda)][0])
            old_min = nLL_lambda[min_ind]
            new_min = vals[max(0,min_ind-1)] + (vals[min_ind] - vals[max(0,min_ind-1)])*0.1
            new_max = vals[min(n-1,min_ind+1)] - (vals[min_ind] - vals[min(n-1,min_ind+1)])*0.1
            vals2 = np.linspace(new_min, new_max, n2)
            nLL_lambda = [0]*n2
            for ind, val in enumerate(vals2):
                parVec = [val, val, 1, 0.0001]
                out = gpr(Xnew, parVec, X1, y1, rbf)

                utilityVec = ucb(out, [beta])
                utilityVec = utilityVec - np.max(utilityVec)

                p = np.exp(utilityVec / tau).tolist()[0]
                p = [max(0.00001,x) for x in p]
                p = p / np.sum(p)
                
                nLL_lambda[ind] = -np.log(p[chosen[i+1]])
            if min(nLL_lambda) < old_min:
                nll[round_index*19 + i] = min(nLL_lambda)
                parameters[round_index*20+i] = vals2[[x for x in range(n2) if nLL_lambda[x]==min(nLL_lambda)][0]]
            else:
                nll[round_index*19 + i] = old_min
                parameters[round_index*20+i] = vals[min_ind]
            
    return nll, parameters
    
def model2(params, subjD, rounds, context):
    return sum(model(params, subjD, rounds, context, 40, 4)[0])

def modelFit(subjD, rounds, context):
    bounds = [(-5,5), (-5,5)]
    fit = sco.differential_evolution(model2, bounds, (subjD, rounds, context), maxiter=200)
    tau, beta = np.exp(fit.x)
    nll, lam = model(xopt, subjD, rounds, context, 40, 4)
    
    return tau, beta, lam, nll
    
def modelFitCMA(subjD, rounds, context):
    opts={'tolx': 1e-2, 'maxfevals': 200, 'verb_log': 0, 'verbose': -9}
    xopt, es = cma.fmin2(model2, [-3,0], 0.5, args=(subjD, rounds, context), options=opts)
    tau, beta = np.exp(xopt)
    nll, lam = model(xopt, subjD, rounds, context, 40, 4)
    
    return tau, beta, lam, nll