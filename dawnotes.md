# how many samples do we have? MCMC models to solve our problems

for fmri problem -- got 15 samples per subj and 10 subj
do I have 150 samples, or 15? major issue -- single subjects can drive effects
so: random effects models e.g. "how many subjects constitute a study" - KFriston

#### MAP ESTIMATION: Shrinkage and empirical priors
new problem:
new subject, simple sample, but have 100's of samples from 100's of subjects,
from same populations --
    e.g. take your individual neuron estimates based on population data

## single subject generative model
could pool data across subjects (fixed effects) -- bad, usaually.
could run individually fit subjects (generative model) -- ok.
hierarchical generative model (random effects) - infer params for pop -- best?

such that generative view --> fit pop vars and subject vars
(where a,b per subject vary between subjects, may vary from generative model)
    need to do MLE of population parameters, integrating over subject a's, b's

## 4 approaches (being careful can shrink your error-bars!)

##### off the shelf software (unlikely)
      - for certain model classes (GLMs, but not RL)
      - linear effects (faculty rating students - which students are best?)
        - can test for bias of females / URM, and can correct fixed effects

##### Summary statistics approach (Holmes and Friston)
      - do low (subject) generative model -- and then estimate pop mean&var
      - alas -- throwing away uncertainty of low level MLE estimates
        - so have both pop variance AND estimation variance
        - but: it cancels out - exactly right variance for group level tests
      - works well for linear regression
      - work poorly for fitting rl models to behavior

##### "expectation maximization"
      - alternative estimating group parameters from individual parameters
      - Quentin Huys 2012 derive the rule (WITH CODE)
      - good get estimates of individual and groups, bad clunky and approx
      - not usually correct to run summary stats on individual estimates
        - error is reduced becaues estimates are dependent
      - nathaniel has julia code doing this

##### Markov Chain Monte Carlo

      - procedure for drawing (correlated) samples for arbitrary model
      - generic packages (Stan, JAGS, BUGS) with fontends for Matlab/Python
      - very general, good scaling
      - gold standard for parameter estimation, tricky for model comparison
      - very different feel from ML, can be finicky, requires monitoring


###### MCMC details

      - super easy to produce simulated data given parameters
      - also easy to produce simulated params from data

      e.g.:

      def model(
        real q[2]
        beta ~ normal(0,10)
        alpha ~ normal (0,1)

        for i in 1:2:
            #generate data from model
            # takes random walk over parameters
            # gibbs sampling for BUGS, Hamiltonian for Stan
            # stan() figures out gradients algebraically, faster, explores better

        )

        - confidence intervals from quantiles of samples
        - sample mean gets closer to true sample mean with more data, but varince stays
        - can also compute samples (and average to obtain expectation) of arbitrary functions of parameters (eg prediction error timeseries [for each set of parameters, so that you can marginalize across generative prediction errors] for fMRI or neural)



###### hierarchy is now trivial

e.g. def model:

        betaM ~ normal(0,10)..
        betaS ~ normal(0,10)..
        alphaM ~ normal(0,1)..
        alphaS ~ normal(0,1)..

          for s in range(0, NS):
            beta[s] = normal(betaM, betaS)
            alpha inv[s] = normal(al
            alpha[s] <- phi approx(alpha_inv[s]); // sigmoid

            for t in range(0, NT):
              ...


###### pragmatics for Stan & MCMC

        - convergence
          - burn in (need to throw out first samples, before convergence)
          - autocorrlation, thinning
          - compare multiple chains, r-hat statistic
          - hairy caterpillar test?!?

        - stan is a big improvement over predecessors
          - last problem: doesn't directly support discrete-valued latent variables (e.g. mixture models); workarounds are laborious
          - BUGS is clumsy, pyMC is appalling, don't roll your own
            (avoid block Metropolis)
        - model comparison is now hard -
        - interdependent parameters make parameter #s hard,
         so BIC for just top params??

        - deep interest (among reviewers) in how evidence for models vary across subjects  -- tricky when you fit simultaneously, b.c. not interdependent
        - alt: compute leave-one-out marginal likelihoods for each subject
          - compare with paired t-tests or submit to SPM_BMS
