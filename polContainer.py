'''
Created on Jan 8, 2013

@author: tiago
'''

import numpy as np
import pycppad as pcad

class PolContainer(object):
	'''
	classdocs
	'''


	def __init__(self):
		'''
		Constructor
		Set default parameters, the user can change them at any time
		by calling setParams
		'''
		params = {}
		params['mu'] = 0.5
		params['w'] = np.array([2.0, 1.0, 2.0])
		params['A'] = np.array([[0,2,0],[5,0,0],[0,3,0]])
		self.setParams(params)
		
	def setParams(self, params):
		'''
		Save parameters in the object and create
		automatic differentiation object
		'''
		self.A  = params['A']
		self.w  = params['w']
		self.mu = params['mu']
		
		ax = pcad.independent(np.zeros(self.w.shape[0]*2))
		ay = self.oscillator(ax)
		self.gf = pcad.adfun(ax, ay)
		self.gf.optimize()
		
	def run(self, x):
		'''
		Return automatically calculated function
		Faster than calling the function in python
		because it gets compiled
		'''
		return self.gf.forward(0, x)
	
	def jacobian(self, x):
		'''
		Return automatically calculated jacobian
		'''
		return self.gf.jacobian(x)
	
	def oscillator(self, x):
		'''
		The actual van der pol oscillator differential equations
		'''
		lx = x.size
		l = lx/2
		res = pcad.ad(np.zeros(np.shape(x)))
		'''position calc \dot{x} = y '''
		res[l:] = x[:l]
		xa = np.tile(x[l:],(l,1))
		dx = xa - xa.transpose()
		'''velocity calc \dot{y} = m*(1-x^2)*y-w^2*x+A.dx'''
		res[:l] = (self.mu*(1-x[l:]*x[l:])*x[:l]-self.w*self.w*x[l:] +
					np.dot(self.A,dx.T).diagonal())
		return res