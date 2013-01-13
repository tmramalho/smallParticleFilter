'''
Created on Jan 8, 2013

@author: tiago
'''

import numpy as np

class FunctionContainer(object):
	'''
	classdocs
	'''


	def __init__(self, params):
		'''
		Constructor
		'''
		self.A  = params['A']
		self.w  = params['w']
		self.mu = params['mu']
		
	def run(self, val):
		l = val.size/2
		res = np.zeros(val.shape)
		'''pos calc \dot{x} = y'''
		res[l:] = val[:l]
		xa = np.tile(val[l:],(l,1))
		dx = xa - xa.transpose()
		'''velocity calc \dot{y} = m*(1-x^2)*y-w^2*x+A.dx'''
		res[:l] = (self.mu*(1-val[l:]*val[l:])*val[:l]-self.w*self.w*val[l:] +
					np.dot(self.A,dx.T).diagonal())
		return res