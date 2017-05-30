#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from scipy.integrate import dblquad
from scipy.interpolate import interp2d

from ProgressBar import progressbar 

class Mesh: 

	def __init__(self, Nx, Ny, xb, yb, a, b, c, q, f0):

		# make variables public 
		self.Nx = Nx 
		self.Ny = Ny 
		self.xb = xb 
		self.yb = yb

		self.f0 = f0 

		self.x, self.y = np.meshgrid(np.linspace(0, xb, Nx+1), np.linspace(0, yb, Ny+1))

		# convert to list of points 
		self.Nnodes = (Nx+1)*(Ny+1)
		print('Number of Nodes =', self.Nnodes)
		self.pts = np.zeros((self.Nnodes, 2))
		ii = 0 
		for i in range(Nx+1):

			for j in range(Ny+1):

				self.pts[ii,0] = self.x[j,i] 
				self.pts[ii,1] = self.y[j,i] 

				ii += 1 

		# create boxes 
		self.ele = [] # store element objects 
		ii = 0 
		for i in range(Nx):

			for j in range(Ny):

				box = np.zeros(4, dtype=int) 

				box[0] = j+(Ny+1)*i
				box[1] = j+(Ny+1)*i+Ny+1
				box[2] = j+(Ny+1)*i+Ny+2
				box[3] = j+(Ny+1)*i+1

				self.ele.append(Element(box, self.pts[box[:],:], ii))

				ii += 1

		# store number of elements used 
		self.Nelements = len(self.ele)

		# determine neighbors 
		for i in range(len(self.ele)):

			for j in range(len(self.ele)):

				# test lower neighbor 
				if (self.ele[i].F[0,0] == self.ele[j].F[2,0] and self.ele[i].F[0,1] == self.ele[j].F[2,1]):

					self.ele[i].neighbor[0] = self.ele[j].elnum

				# test right 
				if (self.ele[i].F[1,0] == self.ele[j].F[3,0] and self.ele[i].F[1,1] == self.ele[j].F[3,1]):

					self.ele[i].neighbor[1] = self.ele[j].elnum

				# test top 
				if (self.ele[i].F[2,0] == self.ele[j].F[0,0] and self.ele[i].F[2,1] == self.ele[j].F[0,1]):

					self.ele[i].neighbor[2] = self.ele[j].elnum

				# test left 
				if (self.ele[i].F[3,0] == self.ele[j].F[1,0] and self.ele[i].F[3,1] == self.ele[j].F[1,1]):

					self.ele[i].neighbor[3] = self.ele[j].elnum

		# determine boundary nodes 
		self.boundary = [] # store bottom, right, top, left boundary node numbers 
		for i in range(4):

			bound = np.array([], dtype=int)
			for el in self.ele:

				if (el.neighbor[i] == -1):

					bound = np.concatenate((bound, el.F[i]))

			self.boundary.append(np.unique(bound))
 
		# generate local stiffness matrices 
		print('generating local stiffness matrices')
		bar = progressbar(len(self.ele), True)
		for el in self.ele:

			el.genStiff(a, b, c, q)

			bar.update()

	def mapScalar(self, f):
		''' map list of values back to grid for plotting ''' 

		grid = np.zeros((self.Ny+1, self.Nx+1))

		ii = 0 
		for i in range(self.Nx+1):

			for j in range(self.Ny+1):

				grid[j,i] = f[ii]

				ii += 1

		np.savetxt('f', grid, delimiter=',')

		return self.x, self.y, grid 

class Element:

	def __init__(self, box, box_pts, elnum):

		self.box = box 

		self.box_pts = box_pts 

		self.elnum = elnum

		self.F = np.zeros((4,2), dtype=int)

		self.F[0,:] = np.array([sorted(self.box[0:2])]) # face 0 
		self.F[1,:] = np.array([sorted(self.box[1:3])]) # face 1 
		self.F[2,:] = np.array([sorted(self.box[2:4])]) # face 2 
		self.F[3,:] = np.array([sorted([self.box[3], self.box[0]])]) # face 3 

		self.neighbor = np.zeros(4, dtype=int) - 1

		self.B = [] # store bases functions 
		self.B.append(lambda xi, eta: .25*(1 - xi)*(1 - eta))
		self.B.append(lambda xi, eta: .25*(1 + xi)*(1 - eta))
		self.B.append(lambda xi, eta: .25*(1 + xi)*(1 + eta))
		self.B.append(lambda xi, eta: .25*(1 - xi)*(1 + eta))

		self.dBxi = [] # store derivatives 
		self.dBxi.append(lambda xi, eta: -.25*(1 - eta))
		self.dBxi.append(lambda xi, eta: .25*(1 - eta))
		self.dBxi.append(lambda xi, eta: .25*(1 + eta))
		self.dBxi.append(lambda xi, eta: -.25*(1 + eta))

		self.dBeta = [] # store derivatives 
		self.dBeta.append(lambda xi, eta: -.25*(1 - xi))
		self.dBeta.append(lambda xi, eta: -.25*(1 + xi))
		self.dBeta.append(lambda xi, eta: .25*(1 + xi))
		self.dBeta.append(lambda xi, eta: .25*(1 - xi))

		self.J = [] 
		self.J.append(lambda xi, eta: 
			self.dBxi[0](xi, eta)*box_pts[0,0] + 
			self.dBxi[1](xi, eta)*box_pts[1,0] + 
			self.dBxi[2](xi, eta)*box_pts[2,0] + 
			self.dBxi[3](xi, eta)*box_pts[3,0])

		self.J.append(lambda xi, eta: 
			self.dBxi[0](xi, eta)*box_pts[0,1] + 
			self.dBxi[1](xi, eta)*box_pts[1,1] + 
			self.dBxi[2](xi, eta)*box_pts[2,1] + 
			self.dBxi[3](xi, eta)*box_pts[3,1])

		self.J.append(lambda xi, eta: 
			self.dBeta[0](xi, eta)*box_pts[0,0] + 
			self.dBeta[1](xi, eta)*box_pts[1,0] + 
			self.dBeta[2](xi, eta)*box_pts[2,0] + 
			self.dBeta[3](xi, eta)*box_pts[3,0])

		self.J.append(lambda xi, eta: 
			self.dBeta[0](xi, eta)*box_pts[0,1] + 
			self.dBeta[1](xi, eta)*box_pts[1,1] + 
			self.dBeta[2](xi, eta)*box_pts[2,1] + 
			self.dBeta[3](xi, eta)*box_pts[3,1])

		self.J_inv = [] # store J inverse 

		# determinant of J 
		self.J_det = lambda xi, eta: self.J[0](xi, eta)*self.J[3](xi, eta) -\
			self.J[2](xi, eta) * self.J[1](xi, eta) 

		# compute inverse 
		self.J_inv.append(lambda xi, eta: 1/self.J_det(xi, eta) * self.J[3](xi, eta))
		self.J_inv.append(lambda xi, eta: -1/self.J_det(xi, eta) * self.J[1](xi, eta))
		self.J_inv.append(lambda xi, eta: -1/self.J_det(xi, eta) * self.J[2](xi, eta))
		self.J_inv.append(lambda xi, eta: 1/self.J_det(xi, eta) * self.J[0](xi, eta))

	def genStiff(self, a, b, c, q):

		self.A = np.zeros((4,4))
		self.rhs = np.zeros(4) 

		for i in range(4): # local node number 

			for j in range(4): # contributions from surrounding local nodes 

				func = lambda xi, eta: (a*self.B[i](xi, eta)*(self.J_inv[0](xi, eta)*self.dBxi[j](xi, eta) 
					+ self.J_inv[1](xi, eta)*self.dBeta[j](xi, eta)) + \
					b*self.B[i](xi, eta)*(self.J_inv[2](xi, eta)*self.dBxi[j](xi, eta) + 
						self.J_inv[3](xi, eta)*self.dBeta[j](xi, eta)) + \
					c*self.B[i](xi, eta)*self.B[j](xi, eta)) * self.J_det(xi, eta)

				integral = dblquad(func, -1, 1, lambda x: -1, lambda x: 1)
				if (integral[1] > 1e-10):

					print('--- WARNING: integral error > 1e-10 --- ')
				self.A[i,j] = integral[0]

				func = lambda xi, eta: self.B[i](xi, eta)*q(self.box_pts[j,0], self.box_pts[j,1])\
					*self.B[j](xi, eta)*self.J_det(xi, eta) 

				integral = dblquad(func, -1, 1, lambda x: -1, lambda x: 1)
				if (integral[1] > 1e-10):

					print('--- WARNING: integral error > 1e-10 --- ')
				self.rhs[i] += integral[0]  

def Assemble(mesh):

	A = np.zeros((mesh.Nnodes, mesh.Nnodes)) # global stiffness matrix 
	b = np.zeros(mesh.Nnodes) # right hand side 

	# begin assembly. loop through elements and assemble global 
	for i in range(mesh.Nelements):

		mEl = mesh.ele[i] # my element (current element)

		for j in range(4): # loop through number of local nodes/element 

			# get global node number = row of matrix 
			node = mEl.box[j] 

			# generate contribution of each local node 
			for k in range(4):

				A[node,mEl.box[k]] += mEl.A[j,k]

			b[node] += mEl.rhs[j]
			
	# apply boundary conditions 
	boundary = np.unique(np.concatenate((mesh.boundary[0], mesh.boundary[3])))

	for mBound in boundary:

		# delete row 
		A[mBound,:] = 0 

		# subtract col 
		b -= mesh.f0*A[:,mBound]

		# remove column 
		A[:,mBound] = 0 

		# set equation to equal RHS value 
		A[mBound, mBound] = 1

		# set boundary value 
		b[mBound] = mesh.f0

	np.savetxt('A.csv', A, delimiter=',')

	# solve for unknowns
	f = np.linalg.solve(A, b)  

	x, y, grid = mesh.mapScalar(f)

	return x, y, grid 

a = 0
b = 1
c = 1 
xb = .1
yb = .1 
Nx = 20
Ny = 20
f0 = 0

f_mms = lambda x, y: np.sin(np.pi*x/xb)*np.sin(np.pi*y/yb)
q = lambda x, y: a*np.pi/xb*np.cos(np.pi*x/xb)*np.sin(np.pi*y/yb) + \
	b*np.pi/yb*np.sin(np.pi*x/xb)*np.cos(np.pi*y/yb) + \
	c*np.sin(np.pi*x/xb)*np.sin(np.pi*y/yb)
# q = lambda x, y: 0 

mesh = Mesh(Nx, Ny, xb, yb, a, b, c, q, f0)

x, y, f = Assemble(mesh)

plt.figure()
plt.pcolor(x, y, f, cmap='viridis')
plt.colorbar()

plt.figure()
for i in range(np.shape(x)[0]):

	plt.plot(x[i,:], f[i,:], label=str(y[i,0]))
plt.title('x')
plt.legend(loc='best')

plt.figure()
for i in range(np.shape(x)[1]):

	plt.plot(y[:,i], f[:,i], label=str(x[0,i]))
plt.title('y')
plt.legend(loc='best')

plt.figure()
plt.pcolor(x, y, np.fabs(f - f_mms(x, y)), cmap='viridis', norm=LogNorm())
plt.colorbar()
plt.title('error')

interp = interp2d(x, y, f)

print('Area =', xb/Nx*yb/Ny)
print('Error =', np.fabs(interp(xb/2, yb/2) - f_mms(xb/2, yb/2)))

plt.show()