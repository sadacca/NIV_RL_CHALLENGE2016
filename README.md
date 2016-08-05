# NIV_RL_CHALLENGE2016
includes: quick python functions for loading and modeling rat reversal task data
          with large master script to do an example model, annotated w/ extra notes

### you will need the following:
- requires: scipy, numpy, and os modules
- also: pandas, bokeh, seaborn, and matplotlib if you want to gussy things up
- this was built in python 3.4, but code should be broadly compatible

### getting started 
- you're going to need to change the directories (hardcoded, sorry) where either rl_import or the master script looks for the *.mat file
- once done -- you should be able to just run the "CHALLENGE2016... .py" for an example of how it all works 
- or you can run "rl_runmain" which calls rl_import and rl_models, and saves the data (roughly 4 min to model 69 sessions 5x each)
- you can then load/plot data with rl_plot -- but note current plots written with bokeh+matplotlib so either those need to be installed or you need to change the code to work with your visualzation library

good luck!
