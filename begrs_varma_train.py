# -*- coding: utf-8 -*-
"""
Created on Mon Aug 30 11:11:24 2021

This script trains a BEGRS surrograte model on simulated VAR/VARMA data.

@author: Sylvain Barde, University of Kent
"""

import pickle
import zlib
import os
import json

from begrs import begrs

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
batchSize = 20000   # Size of training minibatches (same for all)

# Load parametrisations
with open(parametrisationFile) as f:
    json_str = f.read()
    parametrisationDict = json.loads(json_str)

modelTag = parametrisationDict[runModel]['modelTag']
num_latents = parametrisationDict[runModel]['num_tasks']
num_inducing_pts = parametrisationDict[runModel]['num_inducing_pts']
numiter = parametrisationDict[runModel]['numiter']
learning_rate = parametrisationDict[runModel]['learning_rate']
#-----------------------------------------------------------------------------
# Load training samples, parameter ranges and simulated data
fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
datas = zlib.decompress(fil.read(),0)
fil.close()
varData = pickle.loads(datas,encoding="bytes")
parameter_range = varData['parameter_range']
samples = varData['trainSamples']
modelData = varData['trainData']

# Create a begrs estimation object, train on simulated data
begrsEst = begrs()
begrsEst.setTrainingData(modelData, samples, parameter_range)
begrsEst.train(num_latents, num_inducing_pts, batchSize, numiter, 
                learning_rate)

# Save trained model
savePath = modelPath + '/' + modelTag
if not os.path.exists(savePath):
    os.makedirs(savePath,mode=0o777)

lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
begrsEst.save(
    savePath + '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(
        num_inducing_pts,lrnStr,numiter))
#-----------------------------------------------------------------------------
