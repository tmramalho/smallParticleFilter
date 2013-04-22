'''
Created on Jan 8, 2013

@author: tiago
'''

from polContainer import PolContainer
from linearContainer import LinearContainer
from generateTestData import GenerateTestData
from particleFilter import ParticleFilter
from kalmanFilter import KalmanFilter
from extPlotter import extPlot
import numpy as np

if __name__ == '__main__':
	x0 = np.array([0, 1.0, 2.0, 2.0, 0, 0])
	oscillator = PolContainer()
	
	#x0 = np.array([6, 5.4, 6.8])
	#oscillator = LinearContainer()
	dataGen = GenerateTestData(oscillator, x0.size)
	pltWorker = extPlot()
	numSamples = 40
	samplingTime = 10.0/float(numSamples)
	dt = 0.001
	dtf = 0.001
	procs = 0.01
	obs = 0.06
	(x,y) = dataGen.generateSamplePointsGG(samplingTime, numSamples, dt, x0, procs, obs)
	print 'samplePoints generated'
	finalTime = samplingTime*(numSamples-1)
	
	pf = ParticleFilter(oscillator, x0.size)
	pf.runFilter(y, samplingTime, dtf, procs, obs)
	z = pf.getAveragePath()
	pltWorker.plotPath(x, finalTime, 0.5)
	pltWorker.plotMarkers(y, finalTime)
	pltWorker.plotMarkers(z, finalTime, mk='x', ms=8)
	pltWorker.save('sir.pdf')
	pltWorker.clear()
	ac = dataGen.calculateAccuracy(x, z)
	print "Particle filter error:", ac
	
	kf = KalmanFilter(oscillator, x0.size)
	kf.runFilter(y, samplingTime, dtf, procs, obs)
	k = kf.getAveragePath()
	pltWorker.plotPath(x, finalTime, 0.5)
	pltWorker.plotMarkers(y, finalTime)
	pltWorker.plotMarkers(k, finalTime, mk='x', ms=8)
	pltWorker.save('kalman.pdf')
	ac = dataGen.calculateAccuracy(x, k)
	print "Kalman filter error:", ac