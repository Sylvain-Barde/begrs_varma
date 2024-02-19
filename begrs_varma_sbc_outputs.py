# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 09:02:12 2023

This script produces the plots for the Simulated Bayesian Computing diagnostics
on the convergence of the posterior obtained via BEGRS in the VARMA experiment
(figures 2.b, 3.c and 3.d in the paper).

@author: Sylvain Barde, University of Kent
"""

import os
import pickle
import zlib
import json

import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from scipy.stats import binom

# SBC run configurations
save = True
color = True
fontSize = 24
parametrisationFile = 'parametrisations.json'
dataPath = 'simData'
sbcPath = 'sbc'
save_path_figs = 'figures/sbc'

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
for modelList in modelLists:
    #--------------------------------------------------------------------------
    # Load SBC results
    histList = []
    numPts = []
    for runModel in modelList:
        modelTag = parametrisationDict[runModel]['modelTag']
        num_inducing_pts = parametrisationDict[runModel]['num_inducing_pts']
        numiter = parametrisationDict[runModel]['numiter']
        learning_rate = parametrisationDict[runModel]['learning_rate']
        numTasks = parametrisationDict[runModel]['num_tasks']
        varRestrict = parametrisationDict[runModel]['varRestrict']
        
        lrnStr = '{:f}'.format(learning_rate).replace('0.','').rstrip('0')
        fil = open(sbcPath + 
                   '/{:s}_ind_{:d}_lr_{:s}_ep_{:d}.pkl'.format(modelTag,
                                                               num_inducing_pts,
                                                               lrnStr,
                                                               numiter),
                   'rb')
        datas = zlib.decompress(fil.read(),0)
        fil.close()
        sbcData = pickle.loads(datas,encoding="bytes")
        hist_full = sbcData['hist']    
        numPts.append(num_inducing_pts)
    
    
        # Rebin histogram to 50 bins (enough for outputs)
        numReBin = int(hist_full.shape[0]/2)
        hist = np.zeros([numReBin,hist_full.shape[1]])
    
        for i in range(numReBin):
            hist[i,:] = hist_full[2*i,:] + hist_full[2*i+1,:]
    
        histList.append(hist)
    
    #--------------------------------------------------------------------------
    # Load testing data & extract testing series
    fil = open(dataPath + '/' + modelTag + '_data.pkl','rb')
    datas = zlib.decompress(fil.read(),0)
    fil.close()
    varData = pickle.loads(datas,encoding="bytes")
    labels = varData['parameter_labels']
    
    # Generate confidence intervaals based on binomial counts.
    bins = np.arange(histList[0].shape[0])
    numObs = np.sum(histList[0],axis = 0)[0]
    numBins = histList[0].shape[0]
    
    pad = 4;
    confidenceBoundsX = [-0.5-pad,
                         -0.5-pad/2,
                         -0.5-pad,
                         numBins+0.5+pad,
                         numBins+0.5+pad/2,
                         numBins+0.5+pad]
    
    confidenceBoundsY = [binom.ppf(0.005, numObs, 1/numBins),
                         binom.ppf(0.5, numObs, 1/numBins),
                         binom.ppf(0.995, numObs, 1/numBins),
                         binom.ppf(0.995, numObs, 1/numBins),
                         binom.ppf(0.5, numObs, 1/numBins),
                         binom.ppf(0.005, numObs, 1/numBins)]
    
    x_range = max(confidenceBoundsX) - min(confidenceBoundsX)
    xlim_left = min(confidenceBoundsX) - x_range*0.025
    xlim_right = max(confidenceBoundsX) + x_range*0.025
    
    # Plot 
    if varRestrict is True:
        numMat = 1
    else:
        numMat = 2
        
    i = 0 # Counter for list of parameters
    for mat in range(numMat):
        
        subplotSpec = gridspec.GridSpec(ncols=numTasks, nrows=numTasks)
                
        fig = plt.figure(figsize=(16,12))
        for j in range(numTasks):
            for k in range(numTasks):
                subplotCounter = j*numTasks + k
                ax = fig.add_subplot(subplotSpec[subplotCounter])
    
    
                confidenceBounds = ax.fill(confidenceBoundsX,
                                           confidenceBoundsY,
                                           'silver', label = '$95\%$ conf.')
                for hist, numInd, color in zip(histList,numPts,cvec):
                    sbcHist = ax.bar(bins-0.5, 
                                 hist[:,i],
                                 width=1, 
                                 # ec="k", 
                                 align="edge",
                                 edgecolor = None,
                                 color = color, alpha=0.4, 
                                 label = '{:d} ind. pts.'.format(numInd))
    
                # Set y axis limits
                yMin, yMax = ax.axes.get_ylim()
                yMax *= 1.25
    
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
            plt.savefig(save_path_figs + "/sbc_{:s}_{:d}.pdf".format(
                modelTag, mat), format = 'pdf',bbox_inches='tight')