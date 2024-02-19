# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 14:08:27 2023

This script produces the plots for the posterior parameters samples obtained 
with NUTS on the BEGRS surrogate estimation on the VARMA experiment, combined
with the ABC-SMC estimates (figures 2.a, 3.a and 3.b in the paper).

@author: Sylvain Barde, University of Kent
"""
import os
import pickle
import zlib
import json
import pyabc

from matplotlib import pyplot as plt
from matplotlib import gridspec

save = True
color = True
fontSize = 20
parametrisationFile = 'parametrisations.json'
save_path_figs = 'figures/single_estimates'
dataPath = 'simData'
modelPath = 'models'
abcPath = 'abc_smc'
abcIter = 40            # number of ABC-SMC iterations

# Bundle parametrisations together - 3 inducing point settings per config
modelLists = [['mod1', 'mod7',  'mod13'],
              ['mod2', 'mod8',  'mod14'],
              ['mod3', 'mod8',  'mod15'],
              ['mod4', 'mod10', 'mod16'],
              ['mod5', 'mod11', 'mod17'],
              ['mod6', 'mod12', 'mod18']]

with open(parametrisationFile) as f:
    json_str = f.read()
    parametrisationDict = json.loads(json_str)

#-----------------------------------------------------------------------------
# Setup latex output and saving folder
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})

# Setup colors & adapt save folder
if color is True:
    cvec = ['r', 'g','b']
    save_path_figs += '/color'
else:
    cvec = 'k'
    save_path_figs += '/bw'

# Create save folder if required
if save is True:
    if not os.path.exists(save_path_figs):
        os.makedirs(save_path_figs,mode=0o777)

# Iterate over model lists
callsBySetting = []
for modelList in modelLists:
    #--------------------------------------------------------------------------
    # Load results
    flatSamples = []
    numPts = []
    for runModel in modelList:
        modelTag = parametrisationDict[runModel]['modelTag']
        num_inducing_pts = parametrisationDict[runModel]['num_inducing_pts']
        numiter = parametrisationDict[runModel]['numiter']
        learning_rate = parametrisationDict[runModel]['learning_rate']
        numTasks = parametrisationDict[runModel]['num_tasks']
        varRestrict = parametrisationDict[runModel]['varRestrict']
        
        lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
        modelDir = modelPath + '/' + modelTag  + \
           '/begrs_ind_{:d}_lr_{:s}_ep_{:d}'.format(num_inducing_pts,lrnStr,
                                                    numiter)
        
        # Load posterior samples
        fil = open(modelDir + '/estimates.pkl','rb')
        datas = fil.read()
        fil.close()
        results = pickle.loads(datas,encoding="bytes")
        empSamples = results['samples']
        flatSamples.append(empSamples)
        numPts.append(num_inducing_pts)
            
    #--------------------------------------------------------------------------
    # Load testing data & extract testing series
    fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    varData = pickle.loads(datas,encoding="bytes")
    testSamples = varData['testSamples']
    truth = testSamples[0,:]
    labels = varData['parameter_labels']
    parameter_range = varData['parameter_range']
        
    # load SMC estimation
    fil = open(abcPath +'/'+ modelTag + '_SMC_{:d}_iter.pkl'.format(abcIter),
               'rb') 
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    smcDict = pickle.loads(datas,encoding="bytes")
    populations = smcDict['populations']
    population = populations[abcIter-1]
    callsBySetting.append(smcDict['numCalls'])
    #--------------------------------------------------------------------------
    # Plots
    
    if varRestrict is True:
        numMat = 1
    else:
        numMat = 2
    
    i = 0 # Counter for list of parameters
    for mat in range(numMat):
        
        subplotSpec = gridspec.GridSpec(ncols=numTasks, nrows=numTasks)#,
            
        fig = plt.figure(figsize=(16,12))
        for j in range(numTasks):
            for k in range(numTasks):
                subplotCounter = j*numTasks + k
                # d = flat_samples[:,i]
                x_range = parameter_range[i,1] - parameter_range[i,0]
                xlim_left = parameter_range[i,0] - x_range*0.025
                xlim_right = parameter_range[i,1] + x_range*0.025
                
                ax = fig.add_subplot(subplotSpec[subplotCounter])
    
                # plot histogram for BEGRS posterior samples 
                for flatSample, numInd, color in zip(flatSamples,numPts,cvec):
                    d = flatSample[:,i]
                    res = ax.hist(x=d, bins='fd', density = True, 
                                  edgecolor = None, 
                                  color = color, alpha=0.4, 
                                  label = '{:d} ind. pts.'.format(numInd))
                
                # Add ABC-SMC KDE plot
                # !! correct label (early label version used M, not B)
                correctedLabel = labels[i].replace('B','M')
                smc = pyabc.visualization.plot_kde_1d(
                    population['df'],
                    population['w'],
                    xmin = -0.7,
                    xmax = 0.7,
                    x = correctedLabel,
                    xname = correctedLabel,
                    numx = 500,
                    ax = ax,
                    color = 'k',
                    label="ABC-SMC")
                
                # Set y axis limits
                yMin, yMax = ax.axes.get_ylim()
                yMax *= 1.25
                
                # Plot true value
                ax.plot([truth[i],truth[i]], [0, yMax], 'k',  
                        linestyle = 'dashed', 
                        linewidth=1, label = r'Truth')
                
                # Annotate
                plt.text(0.9,0.9, r'${:s}$'.format(labels[i]), 
                         fontsize = fontSize,
                         transform=ax.transAxes)
    
                ax.set_ylabel(''),
                ax.set_xlabel('')
                ax.axes.yaxis.set_ticks([])
            
                ax.set_ylim(top = yMax, bottom = 0)
                ax.set_xlim(left = xlim_left,right = xlim_right)
                ax.plot(xlim_right, 0, ">k", ms=8, clip_on=False)
                ax.plot(xlim_left, yMax, "^k", ms=8, clip_on=False)
                
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_linewidth(2)
                ax.spines['left'].set_linewidth(2)
                ax.tick_params(axis='x', labelsize=fontSize)    
                ax.tick_params(axis='y', labelsize=fontSize)  
                
                # increment counter for next plot
                i += 1
            
        leg = fig.legend(*ax.get_legend_handles_labels(), 
                         loc='lower center', ncol=+len(modelList) + 2,
                         frameon=False, prop={'size':fontSize})
        
        if save is True:
            plt.savefig(save_path_figs + "/dens_{:s}_{:d}.pdf".format(
                modelTag, mat), format = 'pdf',bbox_inches='tight')
    
