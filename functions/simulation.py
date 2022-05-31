import numpy as np
from functions.kernels import rbf
from functions.acquisitionFunctions import ucb
from functions.models import gpr
from IPython.display import clear_output

def gprDF(df, smoothPars, roughPars, replications, roughEnvironments, smoothEnvironments, n=60, n2=6):
    columns = ["trial", "environment", "context", "meanReward", "meanSE"]
    gprdf = {"trial": [x for x2 in range(4) for x in range(20)],
               "environment": (["Smooth"]*20 + ["Rough"]*20)*2,
               "context": ["Conceptual"]*40 + ["Spatial"]*40,
               "meanReward": np.array([]),
               "meanSE": np.array([])}
    smoothTotal = 0
    roughTotal = 0
    choices = [[x1, x2] for x1 in range(8) for x2 in range(8)]
    for task in ["Conceptual", "Spatial"]:
        for x in range(replications):
            # Smooth
            pars = smoothPars[smoothPars["context"]==task]
            par = np.random.randint(0,len(pars))
            tau_s = list(pars["tau"])[par]
            beta_s = list(pars["beta"])[par]
            envNum = np.random.randint(0,40)
            locs = list(df[(df["context"]==task) & (df["trial"]==0) & (df["environment"]==1)]["chosen"])
            location = locs[np.random.randint(0,len(locs))]
            Y = [smoothEnvironments[str(envNum)][str(location)]["y"]*100] 
            y = [[1]]*20
            x1 = [choices[location][0]]
            x2 = [choices[location][1]]
            y[0] = Y[0]
            chosen = [x1[a]*8 + x2[a] for a in range(len(x1))]
            
            for i in range(19):
                X1 = np.column_stack([x1,x2])
                y1 = (np.matrix(Y)-50)/100

                nLL_lambda = [0]*n
                vals = np.linspace(0,10,n)
                vals = vals + vals[1]
                for ind, val in enumerate(vals):
                    parVec = [val, val, 1, 0.0001]
                    out = gpr(choices, parVec, X1, y1, rbf)

                    utilityVec = ucb(out, [beta_s])
                    utilityVec = utilityVec - np.max(utilityVec)

                    p = np.exp(utilityVec / tau_s).tolist()[0]
                    p = p / np.sum(p)

                    nLL_lambda[ind] = -np.log(p[chosen[-1]])
                min_ind = int([x for x in range(n) if nLL_lambda[x]==min(nLL_lambda)][-1])
                old_min = nLL_lambda[min_ind]
                new_min = vals[max(0,min_ind-1)] + (vals[min_ind] - vals[max(0,min_ind-1)])*0.1
                new_max = vals[min(n-1,min_ind+1)] - (vals[min_ind] - vals[min(n-1,min_ind+1)])*0.1
                vals2 = np.linspace(new_min, new_max, n2)
                nLL_lambda = [0]*n2
                for ind, val in enumerate(vals2):
                    parVec = [val, val, 1, 0.0001]
                    out = gpr(choices, parVec, X1, y1, rbf)

                    utilityVec = ucb(out, [beta_s])
                    utilityVec = utilityVec - np.max(utilityVec)

                    p = np.exp(utilityVec / tau_s).tolist()[0]
                    p = p / np.sum(p)
                    
                    nLL_lambda[ind] = -np.log(p[chosen[-1]])
                
                
                if min(nLL_lambda) < old_min:
                    min_lambda = vals2[[x for x in range(n2) if nLL_lambda[x]==min(nLL_lambda)][-1]]
                else:
                    min_lambda = vals[min_ind]
                
                X1 = np.column_stack([x1,x2])
                y1 = (np.matrix(Y)-50)/100
                

                parVec = [min_lambda, min_lambda, 1, 0.0001]
                out = gpr(choices, parVec, X1, y1, rbf)

                utilityVec = ucb(out, [beta_s])
                utilityVec = utilityVec - np.max(utilityVec)
                p = np.exp(utilityVec / tau_s).tolist()[0]
                p = p / np.sum(p)
                location = np.random.choice(range(64), 1, True, p)[0]
                y[i+1] = smoothEnvironments[str(envNum)][str(location)]["y"]*100

                x1 = x1 + [choices[location][0]]
                x2 = x2 + [choices[location][1]]
                Y = Y + [y[i+1]]
                chosen = [x1[a]*8 + x2[a] for a in range(len(x1))]
            if x == 0 and task == "Conceptual":
                smoothTotal = np.array(y)
            else:
                smoothTotal = np.column_stack((smoothTotal, y))

            # Rough
            pars = roughPars[roughPars["context"]==task]
            par = np.random.randint(0,len(pars))
            tau_r = list(pars["tau"])[par]
            beta_r = list(pars["beta"])[par]
            envNum = np.random.randint(0,40)
            locs = list(df[(df["context"]==task) & (df["trial"]==0) & (df["environment"]==0)]["chosen"])
            location = locs[np.random.randint(0,len(locs))]
            Y = [roughEnvironments[str(envNum)][str(location)]["y"]*100] 
            y = [[1]]*20
            x1 = [choices[location][0]]
            x2 = [choices[location][1]]
            y[0] = Y[0]
            chosen = [x1[a]*8 + x2[a] for a in range(len(x1))]
            
            for i in range(19):
                X1 = np.column_stack([x1,x2])
                y1 = (np.matrix(Y)-50)/100
                
                nLL_lambda = [0]*n
                vals = np.linspace(0,10,n)
                vals = vals + vals[1]
                for ind, val in enumerate(vals):
                    parVec = [val, val, 1, 0.0001]
                    out = gpr(choices, parVec, X1, y1, rbf)

                    utilityVec = ucb(out, [beta_r])
                    utilityVec = utilityVec - np.max(utilityVec)

                    p = np.exp(utilityVec / tau_r).tolist()[0]
                    p = p / np.sum(p)

                    nLL_lambda[ind] = -np.log(p[chosen[-1]])
                min_ind = int([x for x in range(n) if nLL_lambda[x]==min(nLL_lambda)][-1])
                old_min = nLL_lambda[min_ind]
                new_min = vals[max(0,min_ind-1)] + (vals[min_ind] - vals[max(0,min_ind-1)])*0.1
                new_max = vals[min(n-1,min_ind+1)] - (vals[min_ind] - vals[min(n-1,min_ind+1)])*0.1
                vals2 = np.linspace(new_min, new_max, n2)
                nLL_lambda = [0]*n2
                for ind, val in enumerate(vals2):
                    parVec = [val, val, 1, 0.0001]
                    out = gpr(choices, parVec, X1, y1, rbf)

                    utilityVec = ucb(out, [beta_r])
                    utilityVec = utilityVec - np.max(utilityVec)

                    p = np.exp(utilityVec / tau_r).tolist()[0]
                    p = p / np.sum(p)
                    
                    nLL_lambda[ind] = -np.log(p[chosen[-1]])
                
                
                if min(nLL_lambda) < old_min:
                    min_lambda = vals2[[x for x in range(n2) if nLL_lambda[x]==min(nLL_lambda)][-1]]
                else:
                    min_lambda = vals[min_ind]
                

                parVec = [min_lambda, min_lambda, 1, 0.0001]
                out = gpr(choices, parVec, X1, y1, rbf)

                utilityVec = ucb(out, [beta_r])
                utilityVec = utilityVec - np.max(utilityVec)
                p = np.exp(utilityVec / tau_r).tolist()[0]
                p = p / np.sum(p)
                location = np.random.choice(range(64), 1, True, p)[0]
                y[i+1] = roughEnvironments[str(envNum)][str(location)]["y"]*100
                x1 = x1 + [choices[location][0]]
                x2 = x2 + [choices[location][1]]
                Y = Y + [y[i+1]]
                chosen = [x1[a]*8 + x2[a] for a in range(len(x1))]
            if x == 0 and task == "Conceptual":
                roughTotal = np.array(y)
            else:
                roughTotal = np.column_stack((roughTotal, y))

            clear_output(wait=True)
            print(task + ": " + str(x+1) + "/" + str(replications))
        if task == "Conceptual":
            gprdf["meanReward"] = np.concatenate((smoothTotal.mean(1), roughTotal.mean(1)))
            gprdf["meanSE"] = np.concatenate((smoothTotal.std(1)/np.sqrt(np.shape(smoothTotal)[0]), roughTotal.std(1)/np.sqrt(np.shape(roughTotal)[0])))
        else:
            gprdf["meanReward"] = np.concatenate((gprdf["meanReward"], smoothTotal.mean(1), roughTotal.mean(1)))
            gprdf["meanSE"] = np.concatenate((gprdf["meanSE"], smoothTotal.std(1)/np.sqrt(np.shape(smoothTotal)[0]), roughTotal.std(1)/np.sqrt(np.shape(roughTotal)[0])))
    return gprdf, smoothTotal, roughTotal