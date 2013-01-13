'''
Created on Jan 8, 2013

@author: tiago
'''

import math
import numpy as np

class GenerateTestData(object):
	'''
	classdocs
	'''


	def __init__(self, function, dimension):
		'''
		Constructor
		'''
		self.func = function
		self.dim = dimension
		
	def generateSamplePointsGG(self, samplingTime, numSamples, dt, x0, procSigma, obsSigma):
		'''
		Get observations for a statespace model defined by
		self.func with gaussian intrinsic noise and
		gaussian observation noise
		'''
		sdt = math.sqrt(dt)
		N = int(samplingTime/float(dt))
		steps = N*(numSamples-1)
		x = np.zeros((steps, self.dim))
		y = np.zeros((numSamples, self.dim))
		noiseAv = np.zeros(self.dim)
		procCov = np.eye(self.dim)*procSigma
		obsCov = np.eye(self.dim)*obsSigma
		xn = np.copy(x0)
		
		for i in xrange(0, numSamples-1):
			y[i] = xn + np.random.multivariate_normal(noiseAv, obsCov)
			for j in xrange(0, N):
				xsi = np.random.multivariate_normal(noiseAv, procCov)
				xn += dt*self.func.run(xn) + sdt*xsi
				x[i*N+j] = np.copy(xn) #check if this is necessary
		y[numSamples-1] = xn + np.random.multivariate_normal(noiseAv, obsCov)
		return (x,y)
	