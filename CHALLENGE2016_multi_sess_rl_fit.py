
########## STARTING and I/O ########################
## you're going to need some functions. import them.
import numpy as np
import scipy.io as io
import scipy.optimize as opti
import scipy.misc as smc
import os
import seaborn as sb
import pandas as pd


## gotta go to where the data is
os.chdir("C:\\users\\sadaccabf\\Downloads")

## gotta load that data
inbound_data = io.loadmat('challenge2016_trainingSet.mat')

## 'aight, so the inbound_data is a 'dict'
## with each variable as an entry
## so let's get those into real vars, shall we?

ratnum = inbound_data["train_rat"]
sesnum = inbound_data["train_sessionnumber"]
events = inbound_data["events"]
data = inbound_data["train_behavior"]

########### DATA OVERVIEW ########################
## now that that's sorted -- what do we have?
##
## ratnum/sesnum are numpy ndarrays.
## because in matlab they were 1d arrays
## you access them like this: ratnum[0,0] for element 1
## ratnum[0,1] for element 2, etc.
## they're 1x69 arrays (rat-by-session)
##
## events/sessions/behavior are python "objects"
## which is a catchall: each element can be int, str, etc.
## which makse sense, in matlab they were cell arrays
## train_beh is a [1xsession(69)][1xtrials(~300)]
## and then it's a dict, because it was a struct in matlab
## with each embedded dict having [7x0] size
## with the struct: [blk/odr/rsp/RT/rwcode/rewT/rew#]
##
## there are several ways to access this data:
## data[0,0]["block"][0,0][0,0] gets you the block for trial1
## data[0,0]["block"][0,8][0,0] gets you the block for trial9
##
## data[0,0][0,0][0][0,0] also gets you the block# for tr1
## data[0,0][0,300][0][0,0] also gets you the block# for tr301
## data[0,0][0,300][2][0,0] gets you the response for tr301


########### IMPORTANT NAN NOTE ######################
## one looming issue is that there be NaNs in the data
## if you care to find them, the answer is to:
## import math
## math.isnan(data[0,0][0,300][4][0,0]) will evaluate to true
## OR:  import numpy
## numpy.isnan(data[0,0][0,300][4][0,0]) will evaluate to true

###### LOOPING THROUGH TRIALS FOR FUN AND PROFIT #########
## OK.  So we can access the data, but how to loop through it?
## easy - find the size you need: e.g. data.size == 69 (nSess)
## and, importantly data[0.0].size = 328 (nTrials)
## so now we can loop through all the data, and avoid NaNs
## to make a loop -- for xx in yy: do zz[xx]
## while keeping an eye on indentation -- no ends here.
##
## sp. ex.1:
##
## sessiontotal = data.size
## for ii in range(0,sessiontotal)
##     trialtotal =datap[0,ii].size
##     for tt in range(0,trialltotal)
##          print(tt) ## disp ALL the 27000 tri #'s
##
## ...and that's it!  get to work you lazy lout!!




########## DEFINE FUNCTIONS WE MAY WANT #####

########### make the main RL function ##########

def rl_function(x0, gamma, data, session):

    alpha = x0[0]
    beta  = x0[1]
    ss    = session

######## INITIALIZING STATE VARIABLES #############
## q_zero - initial choice likelihoods - either general or per state, for each choice
## might need to infer state .. check that out
    num_tot_states = 6; #6 if odorxwell, 2 if just well

    trialtotal = data[0,ss].size
    #q_zero = np.full(trialtotal,num_tot_states-1,1/2)
    #q_state = q_zero[:]
    #c_guess = np.zeros(trialtotal)
    choice = np.zeros(trialtotal)
    #block = np.zeros(trialtotal)
    odor = np.zeros(trialtotal)
    rew_code = np.zeros(trialtotal)
    rew_tim = np.zeros(trialtotal)
    rew_num = np.zeros(trialtotal)
    #react = np.zeros(trialtotal)
    pred_error = np.zeros(trialtotal)
    log_lik = np.zeros(trialtotal)
    q_trial = np.full(num_tot_states,1/2)

###### make the loop#########################

    for tt in range(0,trialtotal):

        ## pull down parameters from data
        odor[tt] = (data[0,ss][0,tt][1][0,0])


        #if there is a choice
        if np.isfinite(data[0,ss][0,tt][2][0,0]):

            ## make the choice
            choice[tt]= data[0,ss][0,tt][2][0,0]
            rew_tim[tt]= data[0,ss][0,tt][5][0,0]
            rew_num[tt]= data[0,ss][0,tt][6][0,0]

            ## OK. what state are we in?  well? wellXodor? wellxblock?
            this_state = np.int16((odor[tt]-1)*2 + choice[tt]-1)
            poss_states = np.int16((odor[tt]-1)*2 + np.array([1, 2])-1)

            log_lik[tt]= beta*q_trial[this_state] - smc.logsumexp(beta*q_trial[poss_states])

            ## if we're in a unsuccessful trial don't update
            ## otherwise, update.
            if np.isnan(rew_code[tt]+rew_tim[tt]+rew_num[tt]):

                expectation = q_trial[this_state]
                observation = 0

            else:

                #get terms for the prediction error
                expectation = q_trial[this_state]
                observation = np.exp(-gamma*(rew_tim[tt]))*rew_num[tt] - q_trial[this_state]

            pred_error[tt] = observation - expectation
            q_trial[this_state] = q_trial[this_state] + alpha*pred_error[tt]

        else:
            pass #q_trial[this_state]


    log_lik = -np.nansum(log_lik)

    return log_lik

#==========================================================
###########################################################

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

    opti_result = opti.minimize(rl_function,x0,args=(gamma, data, session))

    alpha_beta[ii,:]=opti_result["x"]
    #log_lik = rl_function(alpha, beta, gamma, data)
    print(ii/n_sessions)

print(alpha_beta)

## put alphas and betas into a datafram for pretty plots
fit_data = pd.DataFrame(alpha_beta, columns = ['alpha', 'beta'])
sb.regplot(x = 'alpha', y= 'beta', fit_data)
