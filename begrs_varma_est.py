# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 13:25:18 2021

This script estimates a trained BEGRS surrograte model on a single series of 
simulated VAR/VARMA data in order to attempt to recover the known parameters.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import zlib
import json

from begrs import begrs, begrsNutsSampler

#-----------------------------------------------------------------------------
# Define posterior based on prior and surrogate likelihood
def logP(sample):
    
    prior = begrsEst.softLogPrior(sample)
    logL = begrsEst.logP(sample)

    return (prior[0] + logL[0], prior[1] + logL[1])

#-----------------------------------------------------------------------------
# Run configurations
dataPath = 'simData'
modelPath = 'models'
parametrisationFile = 'parametrisations.json'

runModel = 'mod1'   # Pick parameterisation 'mod1' -> 'mod18'
                    # This controls:
                    # - Model version (VAR / VARMA)
                    # - Number of inducing points
                    # - Learning rate & number of training iterations
                    # See 'parametrisations.json' for details

# Load parametrisations
with open(parametrisationFile) as f:
    json_str = f.read()
    parametrisationDict = json.loads(json_str)

# Load BEGRS model
modelTag = parametrisationDict[runModel]['modelTag']
num_inducing_pts = parametrisationDict[runModel]['num_inducing_pts']
numiter = parametrisationDict[runModel]['numiter']
learning_rate = parametrisationDict[runModel]['learning_rate']

lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
modelDir = modelPath + '/' + modelTag  + \
  '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,numiter)
begrsEst = begrs()
begrsEst.load(modelDir)

# Load testing data, get 'empirical' data (first sample)
fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
varData = pickle.loads(datas,encoding="bytes")
testData = varData['testData']
testSamples = varData['testSamples']
xEmp = testData[:,:,0]

# Create & configure NUTS sampler for BEGRS
posteriorSampler = begrsNutsSampler(begrsEst, logP)
init = np.zeros(begrsEst.num_param)
posteriorSampler.setup(xEmp, init)

# Run and save results
N = 10000
burn = 100
posteriorSamples = posteriorSampler.run(N, burn)
sampleESS = posteriorSampler.minESS(posteriorSamples)
print('Minimal sample ESS: {:.2f}'.format(sampleESS))

results = {'truth':testSamples[0,:],
           'mode' : begrsEst.uncenter(posteriorSampler.mode),
           'samples': posteriorSamples,
           'ess': sampleESS}
fil = open(modelDir + '/estimates.pkl','wb')
fil.write(pickle.dumps(results, protocol=2))
fil.close()
#-----------------------------------------------------------------------------