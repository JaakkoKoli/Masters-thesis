import numpy as np
from functions.kernels import rbf
from functions.acquisitionFunctions import ucb
from functions.models import gpr
from IPython.display import clear_output

def gprDF(tau_s, beta_s, tau_r, beta_r, pool, n_grid, replications, roughEnvironments, smoothEnvironments):
    from functions.kernels import rbf
    from functions.acquisitionFunctions import ucb
    from functions.models import gpr
    import numpy as np
    Xnew = [[x, y] for y in range(8) for x in range(8)]
    columns = ["trial", "environment", "context", "meanReward", "meanSE"]
    gprdf = {"trial": [x for x2 in range(4) for x in range(20)],
               "environment": (["Smooth"]*20 + ["Rough"]*20)*2,
               "context": ["Conceptual"]*40 + ["Spatial"]*40,
               "meanReward": np.array([]),
               "meanSE": np.array([])}
    smoothTotal = 0
    roughTotal = 0
    choices = [[x2, x1] for x1 in range(8) for x2 in range(8)]

    for task in ["Conceptual", "Spatial"]:
        for x in range(replications):
            # Smooth
            envNum = np.random.randint(0,40)
            location = np.random.randint(0,64)
            Y = [smoothEnvironments[str(envNum)][str(location)]["y"]*100] 
            y = [1]*20
            x1 = [choices[location][0]]
            x2 = [choices[location][1]]
            y[0] = Y[0]

            def f(n, X, y, Xnew, beta, tau, n_grid, grid_min, grid_max, rbf, ucb, gpr, np):
                val = (n + 1)*((grid_max-grid_min)/n_grid) + grid_min    

                parVec = [val, val, 1, 0.0001]
                out = gpr(Xnew, parVec, X, y, rbf)

                utilityVec = ucb(out, [beta])
                utilityVec = utilityVec - np.max(utilityVec)
                p = np.exp(utilityVec / tau).tolist()[0]
                p = [max(0.00001,x) for x in p]
                p = p / np.sum(p)
                e = 0
                for i in range(len(y)):
                    loc = int(X[i,1]*8 + X[i,0]) # correct?
                    e = e - (np.log(out["sig"][loc]*np.sqrt(2*np.pi)) - 0.5*((y.tolist()[0][i] - out["mu"][loc])/out["sig"][loc])**2)
                return (val, e)
            
            for i in range(19):
                result = pool.map(lambda x: f(x, np.matrix(np.column_stack((x1,x2))), (np.matrix(y[0:i+1])-50)/100, choices, beta_s, tau_s, n_grid, 0, 3, rbf, ucb, gpr, np), range(n_grid))
                min_nll = result[0][1]
                min_ind = 0
                for ind in range(n_grid):
                    if min_nll > result[ind][1]:
                        min_nll = result[ind][1]
                        min_ind = ind

                parVec = [result[min_ind][0], result[min_ind][0], 1, 0.0001]
                out = gpr(Xnew, parVec, np.column_stack((x1,x2)), (np.matrix(y[0:i+1])-50)/100, rbf)

                utilityVec = ucb(out, [beta_s])
                utilityVec = utilityVec - np.max(utilityVec)
                p = np.exp(utilityVec / tau_s).tolist()[0]
                p = [max(0.00001,x) for x in p]
                p = p / np.sum(p)
                location = np.random.choice(range(64), 1, True, p)[0]
                y[i+1] = smoothEnvironments[str(envNum)][str(location)]["y"]*100
                x1 = x1 + [choices[location][0]]
                x2 = x2 + [choices[location][1]]
                Y = Y + [y[i+1]]
            if x == 0 and task == "Conceptual":
                smoothTotal = y
            else:
                smoothTotal = np.column_stack((smoothTotal, y))

            # Rough
            envNum = np.random.randint(0,40)
            location = np.random.randint(0,64)
            Y = [roughEnvironments[str(envNum)][str(location)]["y"]*100] 
            y = [1]*20
            x1 = [choices[location][0]]
            x2 = [choices[location][1]]
            y[0] = Y[0]
            
            for i in range(19):
                result = pool.map(lambda x: f(x, np.matrix(np.column_stack((x1,x2))), (np.matrix(y[0:i+1])-50)/100, choices, beta_r, tau_r, 100, 0, 3, rbf, ucb, gpr, np), range(n_grid))
                min_nll = result[0][1]
                min_ind = 0
                for ind in range(n_grid):
                    if min_nll > result[ind][1]:
                        min_nll = result[ind][1]
                        min_ind = ind

                parVec = [result[min_ind][0], result[min_ind][0], 1, 0.0001]
                out = gpr(Xnew, parVec, np.column_stack((x1,x2)), (np.matrix(y[0:i+1])-50)/100, rbf)

                utilityVec = ucb(out, [beta_s])
                utilityVec = utilityVec - np.max(utilityVec)
                p = np.exp(utilityVec / tau_s).tolist()[0]
                p = [max(0.00001,x) for x in p]
                p = p / np.sum(p)
                location = np.random.choice(range(64), 1, True, p)[0]
                y[i+1] = roughEnvironments[str(envNum)][str(location)]["y"]*100
                x1 = x1 + [choices[location][0]]
                x2 = x2 + [choices[location][1]]
                Y = Y + [y[i+1]]
            if x == 0 and task == "Conceptual":
                roughTotal = y
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
    return gprdf