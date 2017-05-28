#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from genBasis import genBasis 

from scipy.integrate import quadrature as quad 

class FEM: 

	def __init__(self, xe, a, b, q, f0, p, BCL=1):
		''' Arbitrary order Galerkin FEM solver for: 
				a df/dx + b(x) f(x) = q(x)
			with hp mesh refinement 
			Inputs:
				xe: locations of element boundaries 
				a, b, q: equation properties 
					b, q are lambda functions of x 
				f0: boundary value 
				p: polynomial order 
				BCL: sets which boundary is known 
					0: right boundary is known 
					1: left boundary is known 
		''' 

		self.xb = xe[-1] # domain boundary 

		self.xe = xe # element boundary locations 

		self.p = np.zeros(N, dtype=int) + p # initial polynomial order

		self.a = a 
		self.b = b 
		self.q = q
		self.f0 = f0 # initial condition 

		self.BCL = BCL # which side is known boundary  

		self.refine = False # controls mesh refinement 
		self.iter = -1 # store number of refinements 
		self.maxIter = 0 # max number of refinements 

		self.genMesh(self.xe) # generate initial mesh 

	def genMesh(self, xe, PLOT=False):
		''' generate the mesh given the polynomial order and the element boundaries ''' 

		self.N = len(xe) - 1 # number of elements 

		self.x = [] # store node locations 

		self.h = np.zeros(self.N) # element widths 

		for i in range(self.N): 

			self.h[i] = xe[i+1] - xe[i] 

			# compute node locations 
			for j in range(self.p[i]-1):

				self.x.append(xe[i] + self.h[i]/(self.p[i] - 1)*j)

		self.x.append(xb) # add final node location 

		self.x = np.array(self.x) # convert to numpy 

		self.NN = len(self.x) # number of nodes 

		if (PLOT == True):

			for i in range(self.N+1):

				plt.axvline(self.xe[i], ls='--', color='k', alpha=.5)

			for i in range(self.NN):

				plt.axvline(self.x[i], color='b', alpha=.5)

			plt.show()

	def solve(self):

		self.x, f = self.getf() 

		if (self.refine == True):

			while(True):

				# check if maximum refinements reached 
				if (self.iter == self.maxIter):

					print('--- WARNING: maximum number of refinements reached --- ')
					break 

				self.err = self.getErr(f, PLOT=False) # get residual error for each element 

				self.refineMesh(f, self.err) # get new element locations, order 

				# check if need more refinement 
				if (self.refine == False):

					break

				self.x, f = self.getf() # re solve with refined mesh 

				self.iter += 1 # update refinement counter 

				print(self.iter, end='\r')

			print('number of refinements =', self.iter)
			print('final number of elements, nodes =', self.N, self.NN)
			print('min, max, avg p =', np.min(self.p), np.max(self.p), np.mean(self.p))
			print('max, avg residual = {:.6e} {:.6e}'.format(np.max(self.err), np.mean(self.err)))

		return self.x, f

	def getf(self):
		''' solve the FEM equations ''' 

		A = np.zeros((self.NN, self.NN)) # stiffness matrix 

		rhs = np.zeros(self.NN) # right side 

		# loop through elements 
		for i in range(self.N):

			# generate basis functions 
			if (i == 0 or self.p[i] != self.p[i-1]): # only regen if p is changed 

				B, dB = genBasis(self.p[i]) # gen basis functions 

			# loop over nodes in element 
			for j in range(self.p[i]):

				# loop through local bases 
				for k in range(self.p[i]):

					# current node index 
					ind = k + sum(self.p[:i]) - i
					row = j + sum(self.p[:i]) - i

					# gen stiffness coeffs 
					func = lambda xi: self.a*B[j](xi)*dB[k](xi) + \
						self.b(self.x[ind])*B[j](xi)*B[k](xi)*self.h[i]/2 

					# integrate over the element with GQ 
					A[row, ind] += quad(func, -1, 1)[0] 

					# gen rhs 
					func = lambda xi: B[j](xi)*B[k](xi)*self.q(self.x[ind])*self.h[i]/2
					rhs[row] += quad(func, -1, 1)[0] 

		# boundary conditions 
		if (self.BCL == 1): # set left boundary 

			A[0,:] = 0 # remove first equation 
			rhs -= self.f0*A[:,0] # subtract first column 
			A[:,0] = 0 # remove first column 
			A[0,0] = 1 # set first equation to boundary 
			rhs[0] = self.f0 

		else: # set right boundary 

			A[-1,:] = 0 # remove equation 
			rhs -= self.f0*A[:,-1] # subtract last column 
			A[:,-1] = 0 # remove first column 
			# set last equation 
			A[-1,-1] = 1 
			rhs[-1] = self.f0 

		# solve 
		f = np.linalg.solve(A, rhs)

		return self.x, f

	def refineOn(self, tol, maxIter, maxp=10):
		''' Turn hp refinement on 
			Inputs:
				tol: residual tolerance 
				maxIter: max number of refinements
		''' 

		self.refine = True 

		self.tol = tol 
		self.maxIter = maxIter 
		self.maxp = maxp 

	def refineMesh(self, f, err):
		''' halve elements and increase order by one for elements with 
			residual > tol
		''' 

		# build new element location array, change p array 
		xe = [0] # new element locations 
		p = [] # new orders 

		self.refine = False

		# loop through elements 
		for i in range(len(err)):

			# if residual for element i is too large 
			if (err[i] > self.tol):

				self.refine = True # need to rerun 

				# split element in half 
				xe.append(xe[-1] + self.h[i]/2)
				xe.append(xe[-1] + self.h[i]/2)

				# make both halves higher order 
				if (self.p[i] < self.maxp): # increase only if less than maxp 
					
					p.append(self.p[i] + 1)
					p.append(self.p[i] + 1)

				else:

					p.append(self.p[i])
					p.append(self.p[i])

			# if residual is ok 
			else:

				# add normal cell width 
				xe.append(xe[-1] + self.h[i])

				# don't change order 
				p.append(self.p[i])

		# set new xe and p public 
		self.xe = xe 
		self.p = p 

		# generate new mesh 
		self.genMesh(self.xe)

	def getErr(self, f, PLOT=False):
		''' Generate the residual in each element ''' 

		err = np.zeros(self.N) # store residual 

		# loop over elements
		for i in range(self.N):

			# generate basis functions 
			if (i == 0 or self.p[i] != self.p[i-1]): # only regen if p is changed 

				B, dB = genBasis(self.p[i]) # gen basis functions 

			# loop through nodes 
			for j in range(self.p[i]):

				# loop through bases 
				for k in range(self.p[i]):

					ind = k + sum(self.p[:i]) - i
					row = j + sum(self.p[:i]) - i 

					func = lambda xi: self.a*B[j](xi)*dB[k](xi)*f[ind] + \
						self.b(self.x[ind])*B[j](xi)*B[k](xi)*self.h[i]/2*f[ind] - \
						B[j](xi)*B[k](xi)*self.q(self.x[ind])*self.h[i]/2 

					# QG over element 
					err[i] += quad(func, -1, 1)[0] 

		err = np.fabs(err) 

		if (PLOT == True):
			
			plt.semilogy(err, '-o')
			plt.show()

		return err 

	def MMS(self):

		self.f_mms = lambda x: np.sin(np.pi*x/self.xb)

		self.f0 = 0 # set boundary to 0 

		# set q to force solution to f_mms 
		self.q = lambda x: self.a*np.pi/self.xb*np.cos(np.pi*x/self.xb) + \
			self.b(x)*np.sin(np.pi*x/self.xb) 

		x, f = self.solve()

		return x, f, np.linalg.norm(f - self.f_mms(x), 2)/np.linalg.norm(self.f_mms(x), 2)

N = 2 # number of elements 
xb = 5 # boundary 
xe = np.linspace(0, xb, N+1) # element boundaries 
a = 1 
b = lambda x: x<4
q = lambda x: x>2 
f0 = 1 
p = 2

fem = FEM(xe, a, b, q, f0, p)
fem.refineOn(1e-6, 5)
x, f, err = fem.MMS()
# x, f = fem.solve()

plt.plot(x, f, '-o')
plt.show()