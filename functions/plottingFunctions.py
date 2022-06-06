import seaborn 
import numpy as np
import matplotlib.pyplot as plt
from functions.plottingFunctions import *
from functions.kernels import *
from functions.models import *

def createChoiceArray(choices, y, size=8):
    opts = getAllOptions()
    array = np.zeros([size]*2)
    for i, choice in enumerate(choices):
        array[opts[choice]] = y[i]
    return array

def createGPArray(data, gridSize=8):
    array = np.zeros([gridSize,gridSize])
    opts = getAllOptions(gridSize)
    for i, v in enumerate(data):
        array[opts[i][0],opts[i][1]] = v
    return array

def getPost(lam, X, Y):
    opts = [[x, y] for y in range(8) for x in range(8)]
    return gpr(opts, [lam, lam, 1.0, 0.0001], X, Y, rbf)
	
def getId(data, ind):
    return data.iloc[ind*400 + 1]["id"]

def getEnvType(data, ind):
    return data[data["id"]==getId(data, ind)]["environment"].iloc[0]

def getEnvironment(data, rough, smooth, Id, rnd):
    if data[(data["id"] == Id) & (data["round"]==rnd)]["environment"].iloc[0] == 0:
        return rough[str(data[(data["id"] == Id) & (data["round"]==rnd)]["envOrder"].iloc[0])]
    else:
        return smooth[str(data[(data["id"] == Id) & (data["round"]==rnd)]["envOrder"].iloc[0])]

def createEnvironmentArray(environment, gridSize=8):
    array = np.zeros([gridSize]*2)
    for i in range(gridSize**2):
        x = environment[str(i)]
        array[x["x2"], x["x1"]] = x["y"]
    return array
	
def plotTrials(data, df, roughEnvironments, smoothEnvironments, participant, rnd, context, trials):
    roundData = data[(data["context"]==context) & (data["round"]==rnd) & (data["id"]==participant)]
    participantId = getId(df, roundData.iloc[0]["id"])
    
    fig, axes = plt.subplots(len(trials), 3, figsize=(15, len(trials)*5))
    cmap = "YlOrBr"
    for i, trial in enumerate(trials):
        lam = roundData[roundData["trial"]==trial]["lambda"].iloc[0]
        X = df[(df["context"]==context) & (df["id"] == participantId) & (df["round"] == rnd)][["x","y"]][0:trial].astype('float64')
        Y = df[(df["context"]==context) & (df["id"] == participantId) & (df["round"] == rnd)]["z"][0:trial].astype('float64')
        post = getPost(lam, np.matrix(X), np.matrix(Y))
        gpArray = createGPArray(post["mu"])

        choices = df[(df["context"]==context) & (df["id"] == participantId) & (df["round"] == rnd) & (df["trial"] < trial)]["chosen"]
        y = df[(df["context"]==context) & (df["id"] == participantId) & (df["round"] == rnd) & (df["trial"] < trial)]["zscaled"]
        choiceArray = createChoiceArray(np.array(choices), np.array(y))

        env = getEnvironment(df, roughEnvironments, smoothEnvironments, participantId, rnd)
        environmentArray = createEnvironmentArray(env)

        axes[0,0].set_title("Trials", fontsize=16)
        axes[0,1].set_title("Model of participant's belief", fontsize=16)
        axes[0,2].set_title("Environment", fontsize=16)
        seaborn.heatmap(choiceArray, vmin=0, vmax=100, cbar=False, ax=axes[i, 0], cmap=cmap)
        seaborn.heatmap(gpArray, cbar=False, ax=axes[i, 1], cmap=cmap)
        seaborn.heatmap(environmentArray, vmin=0, vmax=1, cbar=False, ax=axes[i, 2], cmap=cmap)


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

    
