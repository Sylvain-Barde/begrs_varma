# -*- coding: utf-8 -*-
"""
Created on Sat Dec  9 10:20:44 2023

This script runs an approximate Bayesian computation with sequential Monte 
Carlo (ABC-SMC) estimation of the VAR(1)/VARMA(1,1) models. This is in order to
provide a reliable comparison benchmark for the BEGRS estimations.

@author: Sylvain Barde, University of Kent
"""

import pickle
import zlib
import os
import tempfile
import pyabc

import numpy as np
from varma_functions import varmaPredict

#------------------------------------------------------------------------------
# Define model and distance functions for ABC-SMC
def model(parameter):
    
    sample = np.asarray([parameter[label] for label in labels])
    
    if varRestrict:
        A = np.asmatrix(
                np.reshape(sample, (num_tasks,num_tasks))
            )
        M = np.asmatrix(np.zeros_like(A))
        
    else:
        A = np.asmatrix(
                np.reshape(sample[0:int(numParam/2)], 
                           (num_tasks,num_tasks))
                )
        M = np.asmatrix(
                np.reshape(sample[int(numParam/2):numParam], 
                           (num_tasks,num_tasks))
                )
        
    simData = varmaPredict(C,A,M,S,truthDataRaw.transpose()).transpose()

    return {"data": simData}

def distance(x, x0):
    return np.mean(abs(x["data"] - x0["data"]))

#------------------------------------------------------------------------------
# Setup of model parametrisations (same as simulation runs)
dataPath = 'simData'
outPath = 'abc_smc'
popSize = 5000
maxIter = 40
plot = False

parametrisationDict = {'mod1':{'modelTag':'var_uncorr',
                                'varRestrict':True,
                                'num_tasks' : 4,
                                'Scorr' : 0},
                        'mod2':{'modelTag':'var_corr_025',
                                'varRestrict':True,
                                'num_tasks' : 4,
                                'Scorr' : 0.25},
                        'mod3':{'modelTag':'var_corr_05',
                                'varRestrict':True,
                                'num_tasks' : 4,
                                'Scorr' : 0.5},
                        'mod4':{'modelTag':'varma_uncorr',
                                'varRestrict':False,
                                'num_tasks' : 3,
                                'Scorr' : 0},
                        'mod5':{'modelTag':'varma_corr_025',
                                'varRestrict':False,
                                'num_tasks' : 3,
                                'Scorr' : 0.25},
                        'mod6':{'modelTag':'varma_corr_05',
                                'varRestrict':False,
                                'num_tasks' : 3,
                                'Scorr' : 0.5}
                        }

if not os.path.exists(outPath):
    os.makedirs(outPath,mode=0o777)

#------------------------------------------------------------------------------
# ABC-SMC extimation
for key, parametrisation in parametrisationDict.items():

    modelTag = parametrisation['modelTag']
    varRestrict = parametrisation['varRestrict']
    num_tasks = parametrisation['num_tasks']
    Scorr = parametrisation['Scorr']
    
    if varRestrict:
        numParam = num_tasks**2
    else:
        numParam = 2*(num_tasks**2)
    
    C = np.asmatrix(np.zeros((num_tasks,1)))
    S = (Scorr*np.asmatrix(np.ones(num_tasks)) 
          + (1-Scorr)*np.asmatrix(np.identity(num_tasks)))
    
    # Load testing data & extract testing series
    fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    varData = pickle.loads(datas,encoding="bytes")
    labels = varData['parameter_labels']
    
    # Get ground truth (test data)
    testData = varData['testData']
    truthDataRaw = testData[:,:,0]
    truthData = {'data':truthDataRaw}
    
    # Get parametrisation of test data
    testSamples = varData['testSamples']
    truthSampleRaw = testSamples[0,:]
    truthSample = dict(zip(labels, truthSampleRaw))    
    
    # Run pyABC estimation
    print(u'\u2500' * 75)
    print('ABC-SMC estimation of: {:s}'.format(modelTag))
    
    prior = pyabc.Distribution(
        **{label : pyabc.RV("uniform", -0.7, 1.4) for label in labels}
        )
    
    abc = pyabc.ABCSMC(model, prior, distance, population_size=popSize)
    db_path = os.path.join(tempfile.gettempdir(), "test.db")
    abc.new("sqlite:///" + db_path, truthData)
    history = abc.run(minimum_epsilon=0.1, max_nr_populations=maxIter)
    
    # Extract populations once run is complete
    numCalls = history.total_nr_simulations
    tMax = history.max_t
    historyDict = {}
    for t in range(tMax+1):
        df, w = history.get_distribution(m=0, t=t)
        historyDict[t] = {'df':df,
                          'w':w}
    
    # Plot if required
    if  plot is True:
    
        pyabc.visualization.plot_kde_matrix(
            historyDict[tMax]['df'],
            historyDict[tMax]['w'])
        
        for i_par, par in enumerate(truthSample.keys()):
        
            pyabc.visualization.plot_kde_1d_highlevel(
                history,
                x=par,
                xname=par,
                xmin=-0.7,
                xmax=0.7,
                numx=500,
                refval=truthSample,
                refval_color="grey",
        )
    
    # Save results
    savedict = {'populations':historyDict,
                'numCalls':numCalls,
                'groundTruth':truthSample}
    fil = open(outPath +'/'+ modelTag + '_SMC_{:d}_iter.pkl'.format(maxIter),
               'wb') 
    fil.write(zlib.compress(pickle.dumps(savedict, protocol=2)))
    fil.close()
