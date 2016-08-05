#
def rl_load()
    ######### STARTING and I/O ########################
    ## you're going to need some functions. import them.
    import scipy.io as io
    import os



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
    os.chdir("C:\\users\\sadaccabf\\NIVCH16")

    return ratnum, sesnum, events, data
