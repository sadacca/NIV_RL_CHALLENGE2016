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

n_sessions = rat_num.size
alpha_beta = np.zeros((n_sessions,2))
n_iter=3


for ss in range(0,n_sessions):
    for ii in range(0,n_iter)
        alpha = np.random.random()
        beta = 5*np.random.random()
        gamma = 4

        x0 = np.array([alpha, beta])
        session = ss

        opti_result = opti.minimize(rl_models.rl_function,x0,args=(gamma, data, session))
        temp_alpha_beta [ii,:] = opti_result["x"]

    alpha_beta[ss,:]=np.median(temp_alpha_beta,0)
    #log_lik = rl_function(alpha, beta, gamma, data)
    print(np.ceil(100*ss/n_sessions))

    label[:,0] = rat_num
    label[:,1] = sess_num

np.save('all_sess_alpha',alpha_beta)
np.save('all_sess_label',label)
