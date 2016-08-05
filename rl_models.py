

########### make the main RL function ##########

def rl_function(x0, data, session):
    import numpy as np
    import scipy.misc as smc


    alpha = x0[0]
    beta  = x0[1]
    gamma = x0[2]
    ss    = session

######## INITIALIZING STATE VARIABLES #############
## q_zero - initial choice likelihoods - either general or per state, for each choice
## might need to infer state .. check that out
    num_tot_states = 6; #6 if odorxwell, 2 if just well

    trialtotal = data[0,ss].size
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




########### make another RL function ##########

def stoopid_mf_rats(x0, data, session):
    import numpy as np
    import scipy.misc as smc


    alpha = x0[0]
    beta  = x0[1]
    gamma = x0[2]
    ss    = session

######## INITIALIZING STATE VARIABLES #############
## q_zero - initial choice likelihoods - either general or per state, for each choice
## might need to infer state .. check that out
    num_tot_states = 2; #6 if odorxwell, 2 if just well

    trialtotal = data[0,ss].size
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
            this_state = np.int16(choice[tt]-1)
            poss_states = np.int16(np.array([1, 2])-1)

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


def multi_well_update_mf_rats(x0, data, session):
    import numpy as np
    import scipy.misc as smc


    alpha = x0[0]
    beta  = x0[1]
    gamma = x0[2]
    ss    = session

######## INITIALIZING STATE VARIABLES #############
## q_zero - initial choice likelihoods - either general or per state, for each choice
## might need to infer state .. check that out
    num_tot_states = 2; #6 if odorxwell, 2 if just well

    trialtotal = data[0,ss].size
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
            this_state = np.int16(choice[tt]-1)
            poss_states = np.int16(np.array([1, 2])-1)

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


def rl_multi_well_update_function(x0, data, session):
    import numpy as np
    import scipy.misc as smc


    alpha = x0[0]
    beta  = x0[1]
    gamma = x0[2]
    ss    = session

######## INITIALIZING STATE VARIABLES #############
## q_zero - initial choice likelihoods - either general or per state, for each choice
## might need to infer state .. check that out
    num_tot_states = 6; #6 if odorxwell, 2 if just well

    trialtotal = data[0,ss].size
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
