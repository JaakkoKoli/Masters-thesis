import seaborn
import numpy as np


def createChoiceArray(choices, size=8):
    opts = getAllOptions()
    array = np.zeros([size]*2)
    for i, choice in enumerate(choices):
        array[opts[choice]] = 1
    return array






#def createArray(data):
#    array = np.zeros([8,8])
#    for i in data:
#        array[data[i]["x1"], data[i]["x2"]] = data[i]["y"]
#    return array

def createAllArrays(data):
    arrays = [0]*len(data)
    for i, j in enumerate(data):
        arrays[i] = createArray(data[j])
    return arrays

def plotHeatmap(array, setRange=True):
    if setRange:
        seaborn.heatmap(array, 0, 1,  annot=True, linewidths=0.5, cbar=False)
    else:
        seaborn.heatmap(array, annot=True, linewidths=0.5, cbar=False)
    plt.plot()

def getAllOptions(gridSize=8):
    return [(x, y) for y in range(gridSize) for x in range(gridSize)]
    
def createArrayGP(data, gridSize=8):
    array = np.zeros([gridSize,gridSize])
    opts = getAllOptions(gridSize)
    for i, v in enumerate(data):
        array[opts[i][0],opts[i][1]] = v
    return array
    
#def createChoiceArray(x,y,z):
#    array = np.zeros([8,8])
#    for i in range(len(x)):
#        array[x[i],y[i]] = z[i]
#    return array

def plotChoices(choices):
    n = len(choices)
    print(n)
    fig, axes = plt.subplots(n, 1, figsize=(15, 100))
    for i in range(n):
        array = createChoiceArray(choices["x"][0:(i+1)],choices["y"][0:(i+1)],choices["z"][0:(i+1)])
        seaborn.heatmap(array, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[i]).set(title="Choice: ["+str(choices["x"][i])+", "+str(choices["y"][i])+"], Value: "+str(choices["z"][i]))
    plt.plot()

def plotResults(choices, environment, out, parameters, p):
    fig, axes = plt.subplots(20, 4, figsize=(20, 100))
    for i in range(20):
        post = createArrayGP(out[i]["mu"]) + 0.5
        array = createChoiceArray(choices["x"][0:(i+1)],choices["y"][0:(i+1)],choices["z"][0:(i+1)])
        seaborn.heatmap(array, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[i, 0]).set(title="Choice: ["+str(choices["x"][i])+", "+str(choices["y"][i])+"], Value: "+str(choices["z"][i]))
        seaborn.heatmap(post, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[i, 1]).set(title="Posterior, lambda: "+str(round(parameters[1,i], 3)))
        seaborn.heatmap(createArrayGP(p[i]), annot=False, linewidths=0.5, cbar=False, ax=axes[i, 3]).set(title="This choice, beta: "+str(round(parameters[0,i], 3)))
        seaborn.heatmap(environment, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[i, 2]).set(title="Ground truth")
    plt.plot()

def plotTrial(i, choices, environment, out, parameters, p, names=["tau", "beta", "lambda"]):
    npars = len(parameters[:,0])
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    post = createArrayGP(out[i]["mu"]) + 0.5
    array = createChoiceArray(choices["x"][0:(i+1)],choices["y"][0:(i+1)],choices["z"][0:(i+1)])
    seaborn.heatmap(array, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[0]).set(title="Choice: ["+str(choices["x"][i])+", "+str(choices["y"][i])+"], Value: "+str(choices["z"][i]))
    seaborn.heatmap(post, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[1]).set(title="Posterior, lambda: "+str(round(parameters[npars-1,i], 3)))
    pars2 = ""
    for ii in range(npars-1):
        pars2 = pars2 + names[ii] + ": " + str(round(parameters[ii,i], 3))
    seaborn.heatmap(createArrayGP(p[i]), annot=True, linewidths=0.5, cbar=False, ax=axes[3]).set(title="This choice, "+pars2)
    seaborn.heatmap(environment, vmin=0, vmax=1, annot=True, linewidths=0.5, cbar=False, ax=axes[2]).set(title="Ground truth")
    seaborn.heatmap(createArrayGP(out[i]["sig"]), annot=True, linewidths=0.5, cbar=False, ax=axes[4]).set(title="sigma")
    plt.plot()
    
def plotAllChoices(choices):
    plotHeatmap(createChoiceArray(choices["x"],choices["y"],choices["z"]))

def plotEnvironment(data):
    plotHeatmap(createArray(data))

def plotLearningCurves(parameters, names=["tau","beta","lambda"]):
    npars = len(parameters[:,0])
    n = len(parameters[0,:])
    fig, axes = plt.subplots(1, npars, figsize=(10, 5))
    for i in range(int(n/20)):
        for ii in range(npars):
            axes[ii].plot(parameters[ii,(i*20):((i+1)*20)])
            axes[ii].set_title(names[ii])
    plt.plot()
    
def getChoices(data, Id, rnd):
    x = df["x"][Id][(rnd*20):((rnd+1)*20)]
    y = df["y"][Id][(rnd*20):((rnd+1)*20)]
    z = [a/100 for a in df["zscaled"][Id][(rnd*20):((rnd+1)*20)]]
    environment = df["environment"][Id][(rnd*20):((rnd+1)*20)][0]
    context = df["context"][Id][(rnd*20):((rnd+1)*20)][0]
    return {"x":x, "y":y, "z":z, "environment":environment, "context":context}
    
def getEnvironment(data, rough, smooth, Id, rnd):
    if data[(data["id"] == Id) & (data["round"]==rnd)]["environment"][0] == 0:
        return createArray(roughEnvironments[str(data[(data["id"] == Id) & (data["round"]==rnd)]["envOrder"])])
    else:
        return createArray(smoothEnvironments[str(data[(data["id"] == Id) & (data["round"]==rnd)]["envOrder"])])