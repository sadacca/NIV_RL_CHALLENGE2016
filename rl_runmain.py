###### IMPORT OUR functions
import numpy as np
import scipy.optimize as opti
import seaborn as sb
import pandas as pd
import rl_import
import rl_models


###### NOW, WE MODEL THE DATA #############################

n_sessions = 69
n_iter=3
alpha_beta = np.zeros((n_sessions,2))

for ii in range(0,n_sessions):

    alpha = np.random.random()
    beta = 5*np.random.random()
    gamma = 4

    x0 = np.array([alpha, beta])
    session = ii

    opti_result = opti.minimize(rl_models.rl_function,x0,args=(gamma, data, session))

    alpha_beta[ii,:]=opti_result["x"]
    #log_lik = rl_function(alpha, beta, gamma, data)
    print(ii/n_sessions)

print(alpha_beta)

## put alphas and betas into a datafram for pretty plots
fit_data = pd.DataFrame(alpha_beta, columns = ['alpha', 'beta'])
sb.regplot('alpha', 'beta', fit_data)
