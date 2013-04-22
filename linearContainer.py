'''
Created on Apr 21, 2013

@author: tiago
'''

import numpy as np

class LinearContainer(object):
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
		params['A'] = np.array([[-0.01,0.8,-0.8], [-0.8,-0.01,0.8], [0.8,-0.8,-0.01]], dtype=float)
		self.setParams(params)
		
	def setParams(self, params):
		'''
		Save parameters in the object
		'''
		self.A  = params['A']
		
	def run(self, x):
		'''
		Return linear function
		'''
		return np.dot(self.A, x)
	
	def jacobian(self, x):
		'''
		Return derivative
		'''
		return self.A