'''
Created on Jan 11, 2013

@author: tiago
'''
from functionContainer import FunctionContainer
from generateTestData import GenerateTestData
from extPlotter import extPlot
import numpy as np

params = {}
params['mu'] = 0.1
params['w'] = np.array([2.0, 1.0, 2.0])
params['A'] = np.array([[0,2,0],[5,0,0],[0,3,0]])
x0 = np.array([0, 1.0, 2.0, 2.0, 0, 0])
	
oscillator = FunctionContainer(params)
	
dataGen = GenerateTestData(oscillator, x0.size)
pltWorker = extPlot()
samplingTime = 1
numSamples = 10
dt = 0.01
(x,y) = dataGen.generateSamplePointsGG(samplingTime, numSamples, dt, x0, 0.1, 0)
print 'samplePoints generated'
	
finalTime = samplingTime*(numSamples-1)