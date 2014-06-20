from numpy import random as r1
from numpy import random as r2
import numpy as np


class newObject(object): pass



params.visitors = newObject()
params.visitors.daily_mean  = 600
params.visitors.daily_stdev = 50

params.experiments = newObject()
params.experiments.n_types  = 2
params.experiments.base_payouts =   np.asarray([0, 0])      # payout in case of no conversion
params.experiments.conv_payouts =   np.asarray([1, 1])       # payout in case of conversion

# This is considered unknown
params.experiments.conv_probabs  = np.asarray([0.3, 0.8])    # probabilities of conversion for each experiment

# number of days to run the compaign for
params.n_days = 20

# params.ab will hold all the parameters for the A/B experiment
params.ab = newObject()
params.ab.test_period = 5
params.ab.initial_probability= np.asarray([0.8, 0.2])




def random_discrete(n_samples, n_possibilities, probabs):
    rand = random.uniform(0, 1, n_samples)
    cum = np.cumsum(probabs)
    output = np.zeros(n_samples)
    for i in range(n_possibilities-1, -1, -1):
        output[rand<=cum[i]] = i
    return np.int32(output)

def get_visitors_for_a_day(probab_experiment):
    """
    Get the 
    """    
    # assume the number of visitors
    n_visitors = int (random.normal(params.visitors.daily_mean, params.visitors.daily_stdev))

    # using the given probabilities, create a vector of the experiments 
    # presented to each visitor
    experiments = random_discrete(n_visitors, params.experiments.n_types, probab_experiment)
    
    # given the experiments, create a vector of the conversion probabilities 
    probab_conversion = params.experiments.conv_probabs[experiments]

    # according to the probability conversions of each visitor, decide 
    # (randomly) if they convert or not
    conversion = np.random.uniform(0, 1, n_visitors) < probab_conversion

    return experiments, conversion
    

def run_AB(params):
    
    daily_payouts = [] * params.n_days
    
    for day in range (params.n_days):
        
        if (day <= params.ab.test_period):        
            experiments, conversion = get_visitors_for_a_day(params.ab.initial_probability)
        else: 
        
        payout = np.zeros(n_visitors)
        avg_payout_per_experiment = np.zeros(params.experiments.n_types)
        for exp in range(params.experiments.n_types):
            payout[np.logical_and(experiments == exp,                conversion) ] = params.experiments.conv_payouts[exp]
            payout[np.logical_and(experiments == exp, np.logical_not(conversion))] = params.experiments.base_payouts[exp]
            avg_payout_per_experiment = np.average(payout[experiments == exp])
        
        day_payout = 
            
        
        
        
run_AB(params)
    
    
    
    
