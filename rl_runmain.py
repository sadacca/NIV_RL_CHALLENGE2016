###### IMPORT OUR functions
import numpy as np
import scipy.optimize as opti
import seaborn as sb
import pandas as pd
import rl_import
import rl_models

###### IMPORT THE DATA ####################################

rat_num, sess_num, events, data = rl_import.rl_load()

###### NOW, WE MODEL THE DATA #############################

alpha_beta = np.zeros((n_sessions,2))
n_sessions = len(rat_num)
#n_iter=3


for ss in range(0,n_sessions):

    alpha = np.random.random()
    beta = 5*np.random.random()
    gamma = 4

    x0 = np.array([alpha, beta])
    session = ss

    opti_result = opti.minimize(rl_models.rl_function,x0,args=(gamma, data, session))

    alpha_beta[ii,:]=opti_result["x"]
    #log_lik = rl_function(alpha, beta, gamma, data)
    print(ii/n_sessions)

np.save('all_sess_alpha',alpha_beta)

## put alphas and betas into a datafram for pretty plots
fit_data = pd.DataFrame(alpha_beta, columns = ['alpha', 'beta'])
sb.regplot('alpha', 'beta', fit_data)
