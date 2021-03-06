from functions.dataProcessing import *
from functions.kernels import *
from functions.models import *
from functions.acquisitionFunctions import *
import scipy.optimize as sco
from multiprocess import Pool
from timeit import default_timer as timer
import cma

def model(params, subjD, rounds, context, n=26, n2=4, track=True):
    start = timer()
    # sample starting values from normal distribution
    tau, beta = np.exp(params)
    Xnew = [[x, y] for y in range(8) for x in range(8)]
    
    # Initialise lists for parameters and other data
    n_rounds = len(rounds)
    n_trials = n_rounds * 20 
    nll = [0]*n_rounds*19
    parameters = np.zeros(19)

    
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
            vals = np.linspace(0,3,n)
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
            if track:
                print("Round "+str(round_index+1)+" -- trial "+str(i+1)+"/19 -- fitting lambda")
            
    if track:
        end = timer()
        print("Fit finished in "+str(round(end - start,2))+"s")
    return nll, parameters

def model2(params, subjD, rounds, context):
    return sum(model(params, subjD, rounds, context, 25, 4, False)[0])

def modelFit(subjD, rounds, context, track=True, opts={'bounds': [-5,5], 'CMA_cmean': 2, 'tolx': 1e-1, 'maxfevals': 50}):
    if track:
        start = timer()
    
    bounds = [(-5,5), (-5,5)]
    #fit = sco.differential_evolution(model2, bounds, (subjD, rounds, context), disp=track, maxiter=200)
    xopt, es = cma.fmin2(model2, [-3,0], 0.5, args=(subjD, rounds, context), options=opts)
    tau, beta = np.exp(xopt)
    nll, lam = model(xopt, subjD, rounds, context, 25, 4, False)
    
    
    if track:
        end = timer()
        print("Fit finished in "+str(round(end - start,2))+"s")
    return tau, beta, lam, nll
    
             
def model_fast(params, subjD, rnd, context, pool, n_grid, track=True):
    from functions.kernels import rbf
    from functions.acquisitionFunctions import ucb
    from functions.models import gpr
    import numpy as np
    start = timer()
    # sample starting values from normal distribution
    #lam = abs(1+np.random.normal())
    tau, beta = np.exp(params)
    Xnew = [[x, y] for y in range(8) for x in range(8)]
    
    # Initialise lists for parameters and other data
    lams = [0]*19
    nlls = [0]*19

    roundD = (np.array(subjD["round"]) == rnd) & (np.array(subjD["context"]) == context)
    chosen = np.array(subjD["chosen"])[roundD]
    y = np.array(subjD["z"])[roundD]
    x1 = np.array(subjD["x"])[roundD]
    x2 = np.array(subjD["y"])[roundD]
    X = np.column_stack((x1,x2))
    n = 19*n_grid

    def f(n, X, y, Xnew, chosen, beta, tau, n_grid, grid_min, grid_max, rbf, ucb, gpr, np):
        i = int((n - (n%n_grid))/n_grid)
        val = (n - (i*n_grid) + 1)*((grid_max-grid_min)/n_grid) + grid_min    

        X1 = np.matrix(X[0:i+1,:])
        y1 = np.matrix(y[0:i+1])

        parVec = [val, val, 1, 0.0001]
        out = gpr(Xnew, parVec, X1, y1, rbf)

        utilityVec = ucb(out, [beta])
        utilityVec = utilityVec - np.max(utilityVec)
        p = np.exp(utilityVec / tau).tolist()[0]
        p = [max(0.00001,x) for x in p]
        p = p / np.sum(p)
        return (i, val, -np.log(p[chosen[i+1]]))
    
    result = pool.map(lambda x: f(x, X, y, Xnew, chosen, beta, tau, n_grid, 0, 3, rbf, ucb, gpr, np), range(n))
    for i in range(19):
        indices = [x for x in range(len(result)) if result[x][0]==i]
        min_nll = result[indices[0]][2]
        min_ind = 0
        for ind in indices:
            if min_nll > result[ind][2]:
                min_nll = result[ind][2]
                min_ind = ind
        lams[i] = result[min_ind][1]
        nlls[i] = min_nll
            
    if track:
        end = timer()
        print("Fit finished in "+str(round(end - start,2))+"s")
    return nlls, lams

def modelFit_fast(subjD, rnd, context, pool, n_grid, track=True):
    if track:
        start = timer()
    
    bounds = [(-5,2), (-5,2)]
    fit = sco.differential_evolution(lambda params: sum(model_fast(params, subjD, rnd, context, pool, n_grid, False)[0]), bounds, disp=track, maxiter=20)
    tau, beta = np.exp(fit.x)
    nlls, lams = model_fast(fit.x, subjD, rnd, context, pool, n_grid, False)
    
    
    if track:
        end = timer()
        print("Fit finished in "+str(round(end - start,2))+"s")
    return tau, beta, lams, nlls
    
def fitMultiple(df, ids, fit_func):   
    res_len = len(ids)*400
    
    nlls = [0]*res_len
    contexts = [0]*res_len
    environments = [0]*res_len
    betas = [0]*res_len
    lambdas = [0]*res_len
    taus = [0]*res_len
    IDs = [0]*res_len
    rnds = [0]*res_len
    trials = [0]*res_len
    chosens = [0]*res_len
    contextOrders = [0]*res_len
    
    cur = 0
    
    for i in ids:
        participant = df.iloc[i]
        for ind in range(20):
            r = participant["round"][ind*20]
            con = participant["context"][ind*20]
            env = participant["environment"][ind*20]
            ord = participant["contextOrder"][ind*20]
            tau, beta, lams, nll = fit_func(participant, [r], con)
            
            for ii in range(20):
                contexts[cur] = con
                environments[cur] = env
                taus[cur] = tau
                betas[cur] = beta
                if ii==0:
                    lambdas[cur] = 0
                    nlls[cur] = 0
                else:
                    lambdas[cur] = lams[ii-1]
                    nlls[cur] = nll[ii-1]
                IDs[cur] = i
                rnds[cur] = r
                trials[cur] = ii
                chosens[cur] = participant["chosen"][ind*20+ii]
                contextOrders[cur] = ord
                cur = cur + 1
                #clear_output(wait=True)
            print(str(round(100*cur/res_len, 1)) + "%")
            
    res = pd.DataFrame({"nll": nlls, 
                        "context": contexts,
                        "environment": environments,
                        "tau": taus,
                        "beta": betas,
                        "lambda": lambdas,
                        "id": IDs, 
                        "round": rnds,
                        "trial": trials,
                        "chosen": chosens, 
                        "contextOrder": contextOrders})
    return res