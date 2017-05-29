#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import dblquad

class Mesh: 

	def __init__(self, Nx, Ny, xb, yb):

		self.x, self.y = np.meshgrid(np.linspace(0, xb, Nx+1), np.linspace(0, yb, Ny+1))

		# convert to list of points 
		self.pts = np.zeros(((Nx+1)*(Ny+1), 2))
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

				box[0] = j+(Nx+1)*i
				box[1] = j+(Nx+1)*i+Ny+1
				box[2] = j+(Nx+1)*i+Ny+2
				box[3] = j+(Nx+1)*i+1
			
				self.ele.append(Element(box, self.pts[box[:],:], ii))

				ii += 1

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

		for i in range(4):

			for j in range(4):

				func = lambda xi, eta: (a*self.B[i](xi, eta)*(self.J_inv[0](xi, eta)*self.dBxi[j](xi, eta) 
					+ self.J_inv[1](xi, eta)*self.dBeta[j](xi, eta)) + \
					b*self.B[i](xi, eta)*(self.J_inv[2](xi, eta)*self.dBxi[j](xi, eta) + 
						self.J_inv[3](xi, eta)*self.dBeta[j](xi, eta)) + \
					c*self.B[i](xi, eta)*self.B[j](xi, eta)) * self.J_det(xi, eta)

				self.A[i,j] = dblquad(func, -1, 1, lambda x: -1, lambda x: 1)[0] 

				func = lambda xi, eta: self.B[i](xi, eta)*q*self.B[j](xi, eta)*self.J_det(xi, eta) 

				self.rhs[i] += dblquad(func, -1, 1, lambda x: -1, lambda x: 1)[0] 

		print(self.rhs)

mesh = Mesh(2, 2, 1, 1)

mesh.ele[0].genStiff(1, 1, 1, 1)