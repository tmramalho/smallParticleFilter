'''
Created on Jun 12, 2012

@author: tiago
'''

import numpy as np

def eulerSolver(time, dt, dim, fun, x0, c0):
	cn = np.copy(c0)
	cp = np.copy(c0)
	xn = np.copy(x0)
	xp = np.copy(x0)
	jt = int(time/float(dt))
	for _ in range(0, jt):
		(f, g) = fun(xp, cp)
		xn = xn + dt*f
		cn = cn + dt*g
	return (xn,cn)

def adamsSolver(time, dt, dim, fun, x0, c0):
	'''Predict the average value for the coupled oscillator'''
	cn = np.copy(c0)
	cp = np.copy(c0)
	xn = np.copy(x0)
	xp = np.copy(x0)
	jt = int(time/float(dt))
	'''adams setup'''
	f = np.zeros((4, dim))
	g = np.zeros((4, dim, dim))
	for i in range(0, 3):
		xp = np.copy(xn)
		cp = np.copy(cn)
		(f[i], g[i]) = fun(xp, cp)
		xs = xp + dt*f[i]
		cs = cp + dt*g[i]
		
		(fs, gs) = fun(xs, cs)
		xn = xp + dt*fs
		cn = cp + dt*gs
		
	for i in range(3, jt):
		xp = np.copy(xn)
		cp = np.copy(cn)
		
		(f[3], g[3]) = fun(xp, cp)
		
		xs = xp + 55*dt/24*f[3] - 59*dt/24*f[2] + 37*dt/24*f[1] - 9*dt/24*f[0]
		cs = cp + 55*dt/24*g[3] - 59*dt/24*g[2] + 37*dt/24*g[1] - 9*dt/24*g[0]
		
		(fs, gs) = fun(xs, cs)
		
		xn = xp + 9*dt/24*fs+19*dt/24*f[3]-5*dt/24*f[2]+dt/24*f[1]
		cn = cp + 9*dt/24*gs+19*dt/24*g[3]-5*dt/24*g[2]+dt/24*g[1]
		
		f = np.roll(f, -1, axis=0)
		g = np.roll(g, -1, axis=0)
	
	return (xn,cn)