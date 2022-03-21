import pandas as pd

def fitOne(df, array_index, fit_func):   
    res_len = 20
    
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
    
    i = int((array_index - (array_index % 20))/20)
    ind = array_index - (i*20)
    
    participant = df.iloc[i]
    r = participant["round"][ind*20]
    con = participant["context"][ind*20]
    env = participant["environment"][ind*20]
    ord = participant["contextOrder"][ind*20]
    tau, beta, lams, nll = fit_func(participant, [r], con)
    
    for ii in range(20):
        contexts[ii] = con
        environments[ii] = env
        taus[ii] = tau
        betas[ii] = beta
        if ii==0:
            lambdas[ii] = 0
            nlls[ii] = 0
        else:
            lambdas[ii] = lams[ii - 1]
            nlls[ii] = nll[ii - 1]
        IDs[ii] = i
        rnds[ii] = r
        trials[ii] = ii
        chosens[ii] = participant["chosen"][ii]
        contextOrders[ii] = ord
            
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
    
def fitOnePart(df, array_index, fit_func):   
    res_len = 400
    
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
    
    i = array_index
    
    participant = df.iloc[i]
    for ind in range(20):
        r = participant["round"][ind*20]
        con = participant["context"][ind*20]
        env = participant["environment"][ind*20]
        ord = participant["contextOrder"][ind*20]
        tau, beta, lams, nll = fit_func(participant, [r], con)
        
        for ii in range(20):
            contexts[ind*20 + ii] = con
            environments[ind*20 + ii] = env
            taus[ind*20 + ii] = tau
            betas[ind*20 + ii] = beta
            if ii==0:
                lambdas[ind*20 + ii] = 0
                nlls[ind*20 + ii] = 0
            else:
                lambdas[ind*20 + ii] = lams[ii - 1]
                nlls[ind*20 + ii] = nll[ii - 1]
            IDs[ind*20 + ii] = i
            rnds[ind*20 + ii] = r
            trials[ind*20 + ii] = ii
            chosens[ind*20 + ii] = participant["chosen"][ind*20 + ii]
            contextOrders[ind*20 + ii] = ord
            
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