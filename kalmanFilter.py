'''
Created on Apr 21, 2013

@author: tiago
'''

import numpy as np
from odeSolver import adamsSolver, eulerSolver

class KalmanFilter(object):
	'''
	classdocs
	'''
	
	def __init__(self, function, dimension):
		'''
		Constructor
		'''
		self.func = function
		self.dim = dimension
		self.I = np.ones(self.dim)
		
	def stateFun(self, x0, C0):
		xp = self.func.run(x0)
		j = self.func.jacobian(x0)
		cp = np.dot(j, C0) + np.dot(C0, j.T) + self.procCov
		return (xp, cp)
		
	def runFilter(self, z, samplingTime, dt, procSigma, obsSigma):
		self.x = []
		self.C = []
		self.procCov = np.eye(self.dim)*procSigma
		self.obsCov = np.eye(self.dim)*obsSigma
		self.numSamples = z.shape[0]
		self.dtype = z.dtype
		if(int(samplingTime/dt) < 10): solver = eulerSolver
		else: solver = adamsSolver
		# estimate position
		self.x.append(z[0])
		# estimate covariance
		self.C.append(self.obsCov+self.procCov)
		for i in xrange(1, self.numSamples):
			xp, P = solver(samplingTime, dt, self.dim, self.stateFun, self.x[i-1], self.C[i-1])
			y = z[i] - xp
			S = P + self.obsCov
			K = np.dot(P, np.linalg.inv(S))
			self.x.append(xp + np.dot(K, y))
			self.C.append(P-np.dot(K, P))
			
	def getAveragePath(self):
		return np.array(self.x)
