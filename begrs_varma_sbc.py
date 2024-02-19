# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 09:43:00 2023

This script runs a simulated Bayesian computing diagnostic on a trained BEGRS
model on The full simulated VAR/VARMA testing data.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import pickle
import zlib
import json

from begrs import begrs, begrsNutsSampler, begrsSbc

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
savePath = 'sbc'

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

# Load SBC testing data
fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
varData = pickle.loads(datas,encoding="bytes")
testData = varData['testData']
testSamples = varData['testSamples']

# Create SBC object, load data & sampler
SBC = begrsSbc()
SBC.setTestData(testSamples,testData)
SBC.setPosteriorSampler(begrsNutsSampler(begrsEst, logP))

# Run and save results
N = 199             # Number of draws
burn = 100          # Burn-in period to discard
init = np.zeros(begrsEst.num_param)
SBC.run(N, burn, init)  # SBC will auto-thin to produce N-burn ranks
SBC.saveData(savePath +
             '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pkl'.format(modelTag,
                                                         num_inducing_pts,
                                                         lrnStr,
                                                         numiter))
#-----------------------------------------------------------------------------
