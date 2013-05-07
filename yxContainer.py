'''
Created on Apr 30, 2013

@author: tiago
'''
import numpy as np

class XYContainer(object):
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
		params['w'] = np.array([0.8,0.1])
		self.setParams(params)
		
	def setParams(self, params):
		'''
		Save parameters in the object
		'''
		self.w  = params['w']
		
	def run(self, x):
		'''
		Return linear function
		'''
		return np.sin(self.w*x)
	
	def jacobian(self, x):
		'''
		Return derivative
		'''
		return self.w*np.cos(self.w*x)