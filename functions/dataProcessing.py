import pandas as pd
import numpy as np
import math 
import json
import seaborn
import matplotlib.pyplot as plt
from datetime import datetime


def readData(path="experimentData/full.csv"):
	rawdata = pd.read_csv(path, sep=",")

	num_rounds = 10
	num_trials = 20
	gridSize_0 = 7

	# Remove empty rows
	rawdata = rawdata[np.logical_and(np.logical_not(rawdata["grid_assignmentID"].isna()), np.logical_not(rawdata["gabor_assignmentID"].isna()))]

	normalise = True
	sampleSize = len(rawdata)
	all_opts = [(x, y) for y in range(gridSize_0+1) for x in range(gridSize_0+1)]
	columns = ["id","age","gender","environment","contextOrder","context","round","trial","x","y","chosen","initx","inity","trajectories","steps","movement","distance","distance_x","distance_y","z","zscaled","previousReward","ts","scale","envOrder","bonus","totalBonus","trajCorrect","trajRMSE","trajAvgSteps","grid_start","grid_end","grid_duration","gabor_start","gabor_end","gabor_duration","grid_gabor_gap","comprehensionTries"]
	df = pd.DataFrame(columns=columns)

	for i in range(sampleSize):
		dat = rawdata.iloc[i,:]
		
		Id = dat["id"]
		age = dat["grid_age"]
		gender = dat["grid_gender"]
		environment = dat["environment"]
		contextOrder = dat["scenario"]
		gridBonus = dat["grid_reward"]
		gaborBonus = dat["gabor_reward"]
		totalBonus = gridBonus + gaborBonus
		
		# Spatial stimuli
		gridHistory = json.loads(str(dat["grid_experimentData"]))
		gridComprehensionTries = gridHistory["comprehensionQuestionTries"]
		grid_x = [x for l in gridHistory["xcollect"] for x in l]
		grid_y = [x for l in gridHistory["ycollect"] for x in l]
		initValues = [x for l in gridHistory["initcollect"] for x in l]
		grid_initx = [x[0] for l in gridHistory["initcollect"] for x in l if x!=None]
		grid_initx = grid_initx[0:195] + [np.nan] + grid_initx[195:199]
		grid_inity = [x[1] for l in gridHistory["initcollect"] for x in l if x!=None]
		grid_inity = grid_inity[0:195] + [np.nan] + grid_inity[195:199]
		grid_chosen = [grid_x[i] + 8*grid_y[i] for i in range(len(grid_x))]
		grid_ts = [x for l in gridHistory["tscollect"] for x in l]
		grid_z = [x for l in gridHistory["zcollect"] for x in l]
		if normalise == True:
			grid_z = [(x-50)/100 for x in grid_z] 
		grid_zscaled = [x for l in gridHistory["zscaledcollect"] for x in l]
		grid_previousz = [np.nan] + grid_z[0:-1]
		gridScale = gridHistory["scaleCollect"][0:num_rounds]
		if 'envOrder' in gridHistory.keys():
			gridEnvOrder = [y for l in [[x]*num_trials for x in gridHistory["envOrder"]] for y in l]
		else:
			gridEnvOrder = [np.nan]*num_rounds*num_trials
		grid_delta_x = abs(np.reshape(grid_x, (num_trials, num_rounds), order="F")[0:num_trials-1] - np.reshape(grid_x, (num_trials, num_rounds), order="F")[1:num_trials])
		grid_delta_y = abs(np.reshape(grid_y, (num_trials, num_rounds), order="F")[0:num_trials-1] - np.reshape(grid_y, (num_trials, num_rounds), order="F")[1:num_trials])
		gridDistance = grid_delta_x + grid_delta_y
		gridDistance = [x for l in np.reshape(np.vstack([[np.nan]*num_rounds, gridDistance]), (num_trials*num_rounds,1), order="F").tolist() for x in l]
		grid_movement_x = abs(np.reshape(grid_x, (num_trials, num_rounds), order="F") - np.reshape(grid_initx, (num_trials, num_rounds), order="F"))
		grid_movement_y = abs(np.reshape(grid_y, (num_trials, num_rounds), order="F") - np.reshape(grid_inity, (num_trials, num_rounds), order="F"))
		gridMovement = [x for l in np.reshape((grid_movement_x + grid_movement_y), (1, num_trials*num_rounds), order="F") for x in l]
		grid_trajectories = [x for l in gridHistory["stepcollect"] for x in l]
		grid_steps = [np.size(x) for x in grid_trajectories]
		gridTraj = gridHistory["trajCollect"]
		gridTrajError = np.sum((np.asarray(gridTraj["targetcollect"])-np.asarray(gridTraj["selectioncollect"]))**2, axis=1)
		gridTrajCorrect = np.sum(gridTrajError == 0) / np.size(gridTrajError)
		gridTrajRMSE = np.sqrt(np.sum(gridTrajError))
		gridTrajAvgSteps = np.mean([len(x) for x in gridTraj["stepcollect"]])
		
		# Conceptual stimuli
		gaborHistory = json.loads(str(dat["gabor_experimentData"]))
		gaborComprehensionTries = gaborHistory["comprehensionQuestionTries"]
		gabor_x = [x for l in gaborHistory["xcollect"] for x in l]
		gabor_y = [x for l in gaborHistory["ycollect"] for x in l]
		initValues = [x for l in gaborHistory["initcollect"] for x in l]
		gabor_initx = [x[0] for l in gaborHistory["initcollect"] for x in l if x!=None]
		gabor_initx = gabor_initx[0:195] + [np.nan] + gabor_initx[195:199]
		gabor_inity = [x[1] for l in gaborHistory["initcollect"] for x in l if x!=None]
		gabor_inity = gabor_inity[0:195] + [np.nan] + gabor_inity[195:199]
		gabor_chosen = [gabor_x[i] + 8*gabor_y[i] for i in range(len(gabor_x))]
		gabor_ts = [x for l in gaborHistory["tscollect"] for x in l]
		gabor_z = [x for l in gaborHistory["zcollect"] for x in l]
		if normalise == True:
			gabor_z = [(x-50)/100 for x in gabor_z] 
		gabor_zscaled = [x for l in gaborHistory["zscaledcollect"] for x in l]
		gabor_previousz = [np.nan] + gabor_z[0:-1]
		gaborScale = gaborHistory["scaleCollect"][0:num_rounds]
		if 'envOrder' in gaborHistory.keys():
			gaborEnvOrder = [y for l in [[x]*num_trials for x in gaborHistory["envOrder"]] for y in l]
		else:
			gaborEnvOrder = [np.nan]*num_rounds*num_trials
		gabor_delta_x = abs(np.reshape(gabor_x, (num_trials, num_rounds), order="F")[0:num_trials-1] - np.reshape(gabor_x, (num_trials, num_rounds), order="F")[1:num_trials])
		gabor_delta_y = abs(np.reshape(gabor_y, (num_trials, num_rounds), order="F")[0:num_trials-1] - np.reshape(gabor_y, (num_trials, num_rounds), order="F")[1:num_trials])
		gaborDistance = gabor_delta_x + gabor_delta_y
		gaborDistance = [x for l in np.reshape(np.vstack([[np.nan]*num_rounds, gaborDistance]), (num_trials*num_rounds,1), order="F").tolist() for x in l]
		gabor_movement_x = abs(np.reshape(gabor_x, (num_trials, num_rounds), order="F") - np.reshape(gabor_initx, (num_trials, num_rounds), order="F"))
		gabor_movement_y = abs(np.reshape(gabor_y, (num_trials, num_rounds), order="F") - np.reshape(gabor_inity, (num_trials, num_rounds), order="F"))
		gaborMovement = [x for l in np.reshape((gabor_movement_x + gabor_movement_y), (1, num_trials*num_rounds), order="F") for x in l]
		gabor_trajectories = [x for l in gaborHistory["stepcollect"] for x in l]
		gabor_steps = [np.size(x) for x in gabor_trajectories]
		gaborTraj = gaborHistory["trajCollect"]
		gaborTrajError = np.sum((np.asarray(gaborTraj["targetcollect"])-np.asarray(gaborTraj["selectioncollect"]))**2, axis=1)
		gaborTrajCorrect = np.sum(gaborTrajError == 0) / np.size(gaborTrajError)
		gaborTrajRMSE = np.sqrt(np.sum(gaborTrajError))
		gaborTrajAvgSteps = np.mean([len(x) for x in gaborTraj["stepcollect"]])
		
		# Round and trial data
		rnd = [y for l in [[x]*num_trials for x in range(num_rounds)] for y in l]*2
		trial = [y for y in [x for x in range(num_trials)]*num_rounds*2]
		n = len(trial) 
		
		# Start and end times
		if type(dat["grid_task_start"]) == str:
			grid_start = datetime.strptime(str(dat["grid_task_start"]),"%Y-%m-%d %H:%M")
			grid_end = datetime.strptime(str(dat["grid_task_end"]),"%Y-%m-%d %H:%M")
			grid_duration = grid_end - grid_start
		else:
			grid_start = np.nan
			grid_end = np.nan
			grid_duration = np.nan
		
		if type(dat["gabor_task_start"]) == str:
			gabor_start = datetime.strptime(str(dat["gabor_task_start"]),"%Y-%m-%d %H:%M")
			gabor_end = datetime.strptime(str(dat["gabor_task_end"]),"%Y-%m-%d %H:%M")
			gabor_duration = gabor_end - gabor_start
		else:
			gabor_start = np.nan
			gabor_end = np.nan
			gabor_duration = np.nan
		
		if type(dat["grid_task_start"]) == str and type(dat["gabor_task_start"]) == str:
			if gabor_start > grid_start:
				grid_gabor_gap = gabor_start - grid_end
			else:
				grid_gabor_gap = grid_start - gabor_end
		else:
			grid_gabor_gap = np.nan
			
		df = df.append({"id": [Id]*n, 
							  "age": [age]*n, 
							  "gender": [gender]*n, 
							  "environment": [environment]*n, 
							  "contextOrder": [contextOrder]*n,
							  "context": ["Spatial"]*(int(n/2)) +  ["Conceptual"]*(int(n/2)),
							  "round": rnd,
							  "trial": trial,
							  "x": grid_x + gabor_x,
							  "y": grid_y + gabor_y,
							  "chosen": grid_chosen + gabor_chosen,
							  "initx": grid_initx + gabor_initx,
							  "inity": grid_inity + gabor_inity,
							  "trajectories": grid_trajectories + gabor_trajectories,
							  "steps": grid_steps + gabor_steps,
							  "movement": gridMovement + gaborMovement,
							  "distance": gridDistance + gaborDistance,
							  "distance_x": np.reshape(np.vstack([[np.nan]*num_rounds, grid_delta_x]), (1, num_trials*num_rounds), order="F")[0].tolist() + np.reshape(np.vstack([[np.nan]*num_rounds, gabor_delta_x]), (1, num_trials*num_rounds), order="F")[0].tolist(),
							  "distance_y": np.reshape(np.vstack([[np.nan]*num_rounds, grid_delta_y]), (1, num_trials*num_rounds), order="F")[0].tolist() + np.reshape(np.vstack([[np.nan]*num_rounds, gabor_delta_y]), (1, num_trials*num_rounds), order="F")[0].tolist(),
							  "z": grid_z + gabor_z,
							  "zscaled": grid_zscaled + gabor_zscaled,
							  "previousReward": grid_previousz + gabor_previousz,
							  "ts": grid_ts + gabor_ts,
							  "scale": [y for l in [[x]*num_trials for x in gridScale] for y in l] + [y for l in [[x]*num_trials for x in gaborScale] for y in l],
							  "envOrder": gridEnvOrder + gaborEnvOrder,
							  "bonus": [gridBonus]*num_rounds*num_trials + [gaborBonus]*num_rounds*num_trials,
							  "totalBonus": [totalBonus]*n,
							  "trajCorrect": [gridTrajCorrect]*(int(n/2)) +  [gaborTrajCorrect]*(int(n/2)),
							  "trajRMSE": [gridTrajRMSE]*(int(n/2)) +  [gaborTrajRMSE]*(int(n/2)),
							  "trajAvgSteps": [gridTrajAvgSteps]*(int(n/2)) +  [gaborTrajAvgSteps]*(int(n/2)),
							  "grid_start": [grid_start]*n,
							  "grid_end": [grid_end]*n,
							  "grid_duration": [grid_duration]*n,
							  "gabor_start": [gabor_start]*n,
							  "gabor_end": [gabor_end]*n,
							  "gabor_duration": [gabor_duration]*n,
							  "grid_gabor_gap": [grid_gabor_gap]*n,
							  "comprehensionTries": [gridComprehensionTries]*(int(n/2)) +  [gaborComprehensionTries]*(int(n/2))
							 }, ignore_index=True)
	return df
	