import numpy as np

def ucb(out, pars, refactor=False, uniqueOptions = 64):
    if refactor:
        gamma = pars[0]
        beta_star = pars[1]
        outtotal = (gamma*out["mu"]) + (beta_star*np.sqrt(out["sig"]))
        outtotal[outtotal < 0] = 0.0001
        outtotal[outtotal > 100] = 100
        d1 = round(np.shape(out)[0]/uniqueOptions)
        return np.reshape(int(np.matrix(outtotal)), (int(d1), int(np.sum(np.shape(outtotal))/d1)), order="F")
    else:
        beta = pars[0]
        outtotal = out["mu"] + (beta*np.sqrt(out["sig"]))
        outtotal[outtotal < 0] = 0.0001
        outtotal[outtotal > 100] = 100
        d1 = round(np.shape(out)[0]/uniqueOptions)
        return np.reshape(np.matrix(outtotal), (int(d1), int(np.sum(np.shape(outtotal))/d1)), order="F")