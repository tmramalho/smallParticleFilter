'''
Created on Jan 9, 2013

@author: tiago
'''

import numpy as np
import math
import scipy.stats as st

class ParticleFilter(object):
	'''
	classdocs
	'''


	def __init__(self, function, dimension):
		'''
		Constructor
		'''
		self.func = function
		self.dim = dimension
		
	def mvGaussian(self, pts, avs, cov):
		invCov = np.linalg.inv(cov)
		av = np.tile(avs, (pts.shape[0], 1))
		'''The following allows us to calculate the dot products
		in the exp part of the gaussian for all pts simultaneously'''
		ip = np.dot(invCov, (pts - av).T).T # C^-1*(av-pt)
		dp = np.diag(np.inner(pts - av, ip)) # (av-pt)^T*C^-1*(av-pt)
		gauss = np.exp(-dp/2)*np.power(2*np.pi, -self.dim*0.5)*np.power(np.linalg.det(cov), -0.5)
		return gauss
		
	def evolvePoints(self, x0):
		xn = np.copy(x0)
		for xp in xn:
			for _ in xrange(0, self.N):
				xsi = np.random.multivariate_normal(self.noiseAv, self.procCov)
				xp += self.dt*self.func.run(xp) + self.sdt*xsi
		return xn
		
	def runFilter(self, y, samplingTime, dt, procSigma, obsSigma, nth = 50, numParticles = 100):
		self.x = []
		self.w = []
		self.dt = dt
		self.sdt = math.sqrt(dt)
		self.N = int(samplingTime/float(dt))
		self.noiseAv = np.zeros(self.dim)
		self.procCov = np.eye(self.dim)*procSigma
		self.obsCov = np.eye(self.dim)*obsSigma
		self.numSamples = y.shape[0]
		self.numParticles = numParticles
		'''for initial guess we use a gaussian around y[0]'''
		xp = np.tile(y[0], (numParticles,1)) + np.random.multivariate_normal(self.noiseAv, self.procCov, self.numParticles)
		wp = self.mvGaussian(xp, y[0], self.obsCov)
		self.x.append(xp)
		self.w.append(wp/np.sum(wp))
		for i in xrange(1, self.numSamples):
			xn = self.resample(xp, wp)
			xp = self.evolvePoints(xn)
			wp = self.mvGaussian(xp, y[i], self.obsCov)
			wp = wp/np.sum(wp)
			neff = 1/np.sum(np.power(wp,2))
			print neff
			self.x.append(xp)
			self.w.append(wp)
			print 'sample', i
		
	def resample(self, xp, wp):
		iprev = range(0, self.numParticles) #particle indexes
		prob = st.rv_discrete(values=(iprev,wp))
		inext = prob.rvs(size = self.numParticles) #new particle indexes
		xr = xp[inext]
		return xr
		
	def getAveragePath(self):
		averages = []
		for i in xrange(0, self.numSamples):
			av = 0
			particles = self.x[i]
			weights = self.w[i]
			for j in xrange(1, weights.size):
				av += particles[j]*weights[j]
			averages.append(av)
		return np.array(averages)
		