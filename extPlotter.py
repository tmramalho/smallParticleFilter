'''
Created on Jan 11, 2012

@author: tiago
'''

import os
import numpy as np
import pylab as plot
import matplotlib.pyplot as plt
import matplotlib.figure as fig
import matplotlib.backends.backend_agg as bkc
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
import matplotlib.colors as col

def gaussian(x, m, s):
	xa = np.zeros((x.shape[0],m.shape[0]))
	xi = np.ones(m.shape)
	for j in range(0, len(xa)):
		xa[j] = np.exp(-(x[j]*xi-m)**2/(2*s))/np.sqrt(2*3.14159265*s)
	return xa

class extPlot():
	'''
	Plots stuff on an external file
	'''

	def __init__(self, wd=None):
		plot.rcParams.update({'font.size': 60})
		plot.rcParams.update({'xtick.major.pad': 20, 'ytick.major.pad': 10})
		plot.rcParams.update({'text.usetex': True})
		plot.rcParams.update({'font.family': 'serif'})
		plot.rcParams.update({'axes.linewidth': 4.0})
		plot.rcParams.update({'xtick.major.width': 4, 'ytick.major.width': 4})
		plot.rcParams.update({'xtick.major.size' : 8, 'ytick.major.size' : 8})
		self.scale = 10
		self.phi = 1.61803399
		self.basefig = fig.Figure(figsize=(self.phi*self.scale, self.scale))
		self.blank =  True
		self.workDir = os.path.expanduser("~") + "/"
		self.vmin = None
		self.vmax = None
		if wd is not None:
			self.workDir = wd
		
	def checkFigureStatus(self, numPlots):
		'''
		Checks if figure has the right size for the number of plots
		If picture is not blank and has right size, the user wants to draw
		stuff on top of what is already there.
		Creates right number of axes and returns the axes object
		'''
		if self.basefig.get_figheight() != numPlots*self.scale or self.blank:
			self.blank = False
			self.basefig.set_figheight(numPlots*self.scale)
			self.basefig.clf()
			for n in range(0, numPlots):
				self.basefig.add_subplot(numPlots, 1, n+1)
			#self.basefig.tight_layout()
		return self.basefig.get_axes()
	
	def clear(self):
		self.basefig.clear()
		self.ax = self.basefig.add_subplot(111)
		self.blank = True
	
	def save(self, filename):
		basecanv = bkc.FigureCanvasAgg(self.basefig)
		basecanv.print_figure(self.workDir + filename,format="pdf",bbox_inches='tight',pad_inches=0.1)
		
	def plotTrajectories(self, av, st, time, dt, a = 1.0, lw = 2.0, cl='map', lim=[0,0]):
		numPlots = len(av[0])
		axes = self.checkFigureStatus(numPlots)
		i = 0
		im = numPlots
		t = np.arange(0, time, dt)
		for ax in axes:
			if cl == 'map':
				col = plt.cm.hsv(float(i)/im,1)
			else:
				col = cl
			l = ax.plot(t, av[:, i])
			plt.setp(l, color=col, ls='-', alpha=a, linewidth=lw)
			l = ax.plot(t, av[:, i]+st[:, i])
			plt.setp(l, color=col, ls='--', alpha=a, linewidth=lw)
			l = ax.plot(t, av[:, i]-st[:, i])
			plt.setp(l, color=col, ls='--', alpha=a, linewidth=lw)
			i += 1
			if lim[0] != lim[1]:
				ax.set_ylim(lim[0], lim[1])
			
	def plotPath(self, av, time, a = 0.2, lw = 2.0, lstyle='-', cl='map'):
		try:
			numPlots = len(av[0])
		except TypeError:
			numPlots = 1
		axes = self.checkFigureStatus(numPlots)
		t = np.linspace(0, time, av.shape[0])
		if numPlots == 1:
			axes[0].plot(t, av)
			return True
		i = 0
		im = numPlots
		for ax in axes:
			if cl == 'map':
				col = plt.cm.Set2(float(i)/im,1)
			else:
				col = cl
			l = ax.plot(t, av[:, i])
			plt.setp(l, color=col, ls=lstyle, alpha=a, linewidth=lw)
			i += 1
			
	def plotMarkers(self, av, time, a = 1.0, mk = 'o', cl='map', ms=4):
		try:
			numPlots = len(av[0])
		except TypeError:
			numPlots = 1
		axes = self.checkFigureStatus(numPlots)
		t = np.linspace(0, time, av.shape[0])
		if numPlots == 1:
			axes[0].plot(t, av)
			return True
		i = 0
		im = numPlots
		for ax in axes:
			if cl == 'map':
				col = plt.cm.Set2(float(i)/im,1)
			else:
				col = cl
			l = ax.plot(t, av[:, i], zorder=100)
			plt.setp(l, color=col, ls='None', marker=mk, alpha=a, markersize=ms)
			i += 1
			
	def plotPFMarkers(self, x, w, time):
		#1st dim is time 2nd numparticles 3rd system dims
		x = np.array(x)
		w = np.array(w)
		try:
			numPlots = x.shape[2]
		except TypeError:
			numPlots = 1
		axes = self.checkFigureStatus(numPlots)
		times = np.linspace(0, time, x.shape[0])
		times = np.repeat(times, x.shape[1])
		times += np.random.rand(times.size)*0.5 - 0.25
		norm = col.Normalize(vmin = np.min(w), vmax = np.max(w))
		weights = plt.cm.jet(norm(np.ravel(w)), 0.5)
		for i,ax in enumerate(axes):
			ax.scatter(times, np.ravel(x[:, :, i]), c=weights)
		
	def plotHistogramPoints(self, x):
		axes = self.checkFigureStatus(1)
		n, self.bins, patches = axes[0].hist(x, 40, normed=True)
		i = 0
		im = len(x[0])
		for p in patches: #iterate all dimensions
			for artist in p: #iterate all the rectangles
				artist.set_color(plt.cm.hsv(float(i)/im,1))
			i += 1
		
	def plotHistogramPrediction(self, m, s, a = 1.0, lw = 2.0):
		axes = self.checkFigureStatus(1)
		lines = axes[0].plot(self.bins, gaussian(self.bins, m, s))
		i = 0
		im = len(m)
		for l in lines:
			plt.setp(l, color=cm.hsv(float(i)/im,1), ls='-', alpha=a, linewidth=lw)
			i += 1
	
	def plotSingleHistogramPoints(self, x, dim):
		df = float(dim)
		axes = self.checkFigureStatus(1)
		n, self.bins, patches = axes[0].hist(x, 100, normed=True)
		t = np.arange(0, np.max(x), np.max(x)/1000)
		l = axes[0].plot(t, 1 / (2*sp.gamma(df/2)) * np.power(t/2,df/2-1) * np.exp(-t/2))
		plt.setp(l, linewidth=2)
		axes[0].set_xlim(0,16) #TODO: this should not be hardcoded
		
		
	def plotHeatMap(self, c):
		axes = self.checkFigureStatus(1)
		if self.vmin is None:
			self.vmin = np.min(c[::-1])
			self.vmax = np.max(c[::-1])
		#image = axes[0].imshow(c, interpolation='nearest')
		#axes[0].grid(True)
		image = axes[0].pcolormesh(c[::-1], edgecolors='k', cmap=cm.RdYlBu, vmin=self.vmin, vmax=self.vmax)
		self.basefig.colorbar(image)
		axes[0].get_xaxis().set_ticks(np.arange(1,len(c[0])+1))
		axes[0].get_xaxis().set_ticks_position('top')
		axes[0].get_yaxis().set_ticks(np.arange(len(c[0]),0,-1))
		
	def plotPaperRawTrajectories(self, x, time, dt):
		index = 0 #hardcoded
		numPlots = 3
		axes = self.checkFigureStatus(numPlots)
		t = np.arange(0, time, dt)
		i = 0
		for ax in axes:
			if i == 0:
				l = ax.plot(t, x[:, index])
				plt.setp(l, ls="-", alpha=0.2, linewidth=1, color='k')
			i += 1
		
	def plotPaperTrajectories(self, mcAverage, mcStdDev, simAverage, simStdDev, linearAverage, linearStdDev, time, dt):
		index = 0 #hardcoded
		numPlots = 3
		axes = self.checkFigureStatus(numPlots)
		i = 0
		im = numPlots
		t = np.arange(0, time, dt)
		for ax in axes:
			if i == 0:
				l = ax.plot(t, mcAverage[:, index])
				plt.setp(l, ls='-', alpha=1, linewidth=4, color='k')
				self.paperLegendAdjustments(ax, 5, r"$x_0$")
			elif i == 1:
				l = ax.plot(t, mcAverage[:, index])
				plt.setp(l, ls='-', alpha=1, linewidth=4, color='k')
				l = ax.plot(t, simAverage[:, index])
				l[0].set_dashes([40, 20])
				plt.setp(l, ls='--', alpha=1, linewidth=4, color='k')
				l = ax.plot(t, linearAverage[:, index])
				plt.setp(l, ls='-', alpha=1, linewidth=4, color='k')
				l[0].set_dashes([4, 18])
				self.paperLegendAdjustments(ax, 5, r"$x_0$")
			elif i == 2:
				l = ax.plot(t, mcStdDev[:, index])
				plt.setp(l, ls='-', alpha=1, linewidth=4, color='k')
				l = ax.plot(t, simStdDev[:, index])
				plt.setp(l, ls='--', alpha=1, linewidth=4, color='k')
				l[0].set_dashes([40, 20])
				l = ax.plot(t, linearStdDev[:, index])
				plt.setp(l, ls='-', alpha=1, linewidth=4, color='k')
				l[0].set_dashes([4, 10])
				self.paperLegendAdjustments(ax, 6, r"$\sigma_{x_0}$")
			i += 1
			
	def paperLegendAdjustments(self, ax, numTicks, text):
		ax.set_ylabel(text)
		ax.set_xlabel(r"$t$")
		ticks = ax.get_yticks()
		newTicks = np.around(np.linspace(min(ticks), max(ticks), 5), 1)
		ax.set_yticks(newTicks)
		ax.yaxis.set_label_coords(-0.15, 0.5)
		
	def paperFullLegendAdjustments(self, ax, numXTicks, xLabel, numYTicks, yLabel):
		ax.set_ylabel(yLabel)
		ax.set_xlabel(xLabel)
		ticks = ax.get_yticks()
		newTicks = np.around(np.linspace(min(ticks), max(ticks), numYTicks), 1)
		ax.set_yticks(newTicks)
		ax.yaxis.set_label_coords(-0.15, 0.5)
		ticks = ax.get_xticks()
		newTicks = np.around(np.linspace(min(ticks), max(ticks), numXTicks), 1)
		ax.set_xticks(newTicks)
		ax.xaxis.set_label_coords(0.5, -0.15)
			
	def plotPaperHistogramPoints(self, x):
		axes = self.checkFigureStatus(4)
		self.bins = {}
		for i in range(0,3):
			n, self.bins[i], patches = axes[i].hist(x[:,i], 40, normed=True)
			for p in patches: #iterate all dimensions
				try:
					for artist in p: #iterate all the rectangles
						artist.set_color('k')
						artist.set_alpha(0.4)
				except TypeError:
					p.set_color('k')
					p.set_alpha(0.4)
		
		
	def plotPaperHistogramPrediction(self, m, s, a = 1.0, lw = 2.0):
		axes = self.checkFigureStatus(4)
		for i in range(0,3):
			g = gaussian(self.bins[i], m, s)
			lines = axes[i].plot(self.bins[i], g[:,i])
			plt.setp(lines, ls='-', alpha=1, linewidth=3, color='k')
			self.paperFullLegendAdjustments(axes[i], 6, r'$x_'+str(i)+'$', 6, r'$P(x_'+str(i)+')$')
	
	def plotPaperSingleHistogramPoints(self, x, dim):
		df = float(dim)
		axes = self.checkFigureStatus(4)
		n, self.bins, patches = axes[3].hist(x, 100, normed=True)
		for p in patches: #iterate all dimensions
			try:
				for artist in p: #iterate all the rectangles
					artist.set_color('k')
			except TypeError:
				p.set_color('k')
				p.set_alpha(0.4)
		t = np.arange(0, np.max(x), np.max(x)/1000)
		l = axes[3].plot(t, 1 / (2*sp.gamma(df/2)) * np.power(t/2,df/2-1) * np.exp(-t/2))
		plt.setp(l, ls='-', alpha=1, linewidth=3, color='k')
		self.paperFullLegendAdjustments(axes[3], 6, r"$\mathcal{X}^2$", 6, r'$P(\mathcal{X}^2)$')
