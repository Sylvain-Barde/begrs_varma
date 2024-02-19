# -*- coding: utf-8 -*-
"""
Created on Tue Feb 23 15:54:48 2021

This script produces the simulated data required to run the VAR/VARMA analysis
in section 4.1 of the paper.

@author: Sylvain Barde, University of Kent
"""

import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import os
import zlib

from varma_functions import get_sobol_samples, varmaSim

sim_path = "simData"                    # path to save simulated data
out_path = "figures/varma_stability"    # path to save eigenvalue plots

# Setup figure output parameters
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"]})
ax_min = -1.2
ax_max = 1.2
fontSize = 40

# Parametrisation of VARMA simulations
skip_init = 500             # number of initial sobol draws to skip
bufferRatio = 1.5           # size of raw sobol draw relative to sample size
numSamplesList = [1000,     # Number of training/testing samples
                  1000]
N = 200                     # Number of effective observations per series
burn = 50                   # Number of burn-in obvservtions

parametrisationDict = {
    'mod1':{'modelTag':'var_uncorr',
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

#------------------------------------------------------------------------------
# Iterate over VARMA parametrisations
if not os.path.exists(sim_path):
    os.makedirs(sim_path,mode=0o777)
if not os.path.exists(out_path):
    os.makedirs(out_path,mode=0o777)

skips = [skip_init, skip_init + int(numSamplesList[0]*bufferRatio)]
tags = ['Training data','Testing data']
shortTags = ['train','test']

for key, parametrisation in parametrisationDict.items():
    modelTag = parametrisation['modelTag']
    varRestrict = parametrisation['varRestrict']
    num_tasks = parametrisation['num_tasks']
    Scorr = parametrisation['Scorr']
        
    if varRestrict:
        numParam = num_tasks**2
        A_Label = r'A_{%(indI)d,%(indJ)d}'
        paramLabels = [A_Label % {'indI' : i+1, 'indJ' : j+1} 
                for i in range(num_tasks) for j in range(num_tasks)]
        
    else:
        numParam = 2*(num_tasks**2)
        A_Label = r'A_{%(indI)d,%(indJ)d}'
        M_Label = r'B_{%(indI)d,%(indJ)d}'
        A_labels = [A_Label % {'indI' : i+1, 'indJ' : j+1} 
                for i in range(num_tasks) for j in range(num_tasks)]
        M_labels = [M_Label % {'indI' : i+1, 'indJ' : j+1} 
                for i in range(num_tasks) for j in range(num_tasks)]
        paramLabels = A_labels + M_labels
    
    # Initialise output dict
    parameter_range = np.array([(-0.7, 0.7)]*numParam)
    results = {'parameter_range':parameter_range,
               'parameter_labels':paramLabels}
    
    #--------------------------------------------------------------------------
    # Simulate VARMA training/testing data
    rng_seed = 0
    np.random.seed(rng_seed)
   
    print(u'\u2500' * 75)
    print('Simulating: {:s}'.format(modelTag))
    
    for numSamples, skip, tag, shortTag in zip(numSamplesList,
                                                skips,
                                                tags,
                                                shortTags):
    
        #----------------------------------------------------------------------
        # Testing for stable VARMA parametrisations
        samples_raw = get_sobol_samples(int(numSamples*bufferRatio), 
                                        parameter_range,skip)
        
        samplesEigen = []
        samples = []
        MaxL_norms = []
        
        for count, sample in enumerate(samples_raw):
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
            
            L_a,U_a = np.linalg.eig(A)
            L_a_norm = np.absolute(L_a)
            
            L_m,U_m = np.linalg.eig(M)
            L_m_norm = np.absolute(L_m)
            
            if max(L_a_norm) < 1 and max(L_m_norm) < 1:
                samplesEigen.append([L_a,L_m])
                samples.append(sample)
                MaxL_norms.append([max(L_a_norm),max(L_m_norm)])
                
            if len(samples) == numSamples:
                samplesEigen = np.asarray(samplesEigen)
                samples = np.asarray(samples)
                break   
        
        results['{:s}Samples'.format(shortTag)] = samples
        print('{:s}:'.format(tag))
        print(' {:d} draws required for {:d} samples'.format(count+1,
                                                             numSamples))
        
        #----------------------------------------------------------------------
        # Simulate VARMA training data
        C = np.asmatrix(np.zeros((num_tasks,1)))
        S = (Scorr*np.asmatrix(np.ones(num_tasks)) 
              + (1-Scorr)*np.asmatrix(np.identity(num_tasks)))
        simData = np.zeros([N,num_tasks,numSamples])
        for i,sample in enumerate(samples):
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
            simData[:,:,i] = varmaSim(C,A,M,S,N,burn).transpose()
         
        results['{:s}Data'.format(shortTag)] = simData
        
        #----------------------------------------------------------------------    
        # Plot eigenvalues
        pi_vec = np.arange(0,2*math.pi,(2*math.pi)/1000)
        figLabels = ['A','M']
        for i in range(samplesEigen.shape[1]):
            eigs = samplesEigen[:,i,:].flatten()
            
            fig, ax = plt.subplots(figsize=(12,12))
            ax.plot(np.cos(pi_vec),np.sin(pi_vec),'k', linewidth=1)
            ax.scatter(np.real(eigs), np.imag(eigs), color = [0.8, 0.8, 0.8])
            ax.set_ylim(top = ax_max, bottom = ax_min)
            ax.set_xlim(left = ax_min,right = ax_max)
            ax.plot(ax_max, 0, ">k", clip_on=False)
            ax.plot(0, ax_max, "^k", clip_on=False)
            ax.set_aspect('equal', 'box')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_linewidth(2)
            ax.spines['bottom'].set_position('zero')
            ax.spines['left'].set_linewidth(2)
            ax.spines['left'].set_position('zero')
            ax.set_xticks([-1,1])
            ax.set_xticklabels(['',''])
            ax.set_yticks([-1,1])
            ax.set_yticklabels(['',''])
            ax.xaxis.set_tick_params(length = 10, width = 2)
            ax.yaxis.set_tick_params(length = 10, width = 2)
            ax.annotate('1', xy=(1.05, -0.1), xytext=(1.05, -0.15), 
                        size = fontSize)
            ax.annotate('-1', xy=(-1.1, -0.1), xytext=(-1.15, -0.15), 
                        size = fontSize)
            ax.annotate('$i$', xy=(-0.1, 1.05), xytext=(-0.1, 1.05), 
                        size = fontSize)
            ax.annotate('$-i$', xy=(-0.15, -1.1), xytext=(-0.2, -1.1), 
                        size = fontSize)
            
            plt.savefig(out_path + '/{:s}_stability_{:s}_{:s}.pdf'.format(
                                                              modelTag,
                                                              shortTag,
                                                              figLabels[i]), 
                        format = 'pdf') 
    
    #--------------------------------------------------------------------------
    # Save samples and simulations
    fil = open(sim_path + '/' + modelTag + '_data.pkl','wb') 
    fil.write(zlib.compress(pickle.dumps(results, protocol=2)))
    fil.close()
