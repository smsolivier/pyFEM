#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from scipy.integrate import quadrature as quad 

class Mesh: 

	def __init__(self, Nx, Ny, xb, yb):

		x, y = np.meshgrid(np.linspace(0, xb, Nx+1), np.linspace(0, yb, Ny+1))

		# convert to list of points 
		pts = np.zeros(((Nx+1)*(Ny+1), 2))
		ii = 0 
		for i in range(Nx+1):

			for j in range(Ny+1):

				pts[ii,0] = x[j,i] 
				pts[ii,1] = y[j,i] 

				ii += 1 

		# create boxes 
		ele = [] # store element objects 
		ii = 0 
		for i in range(Nx):

			for j in range(Ny):

				box = np.zeros(4, dtype=int) 

				box[0] = j+(Nx+1)*i
				box[1] = j+(Nx+1)*i+Ny+1
				box[2] = j+(Nx+1)*i+Ny+2
				box[3] = j+(Nx+1)*i+1
			
				ele.append(Element(box, pts[box[:],:], ii))

				ii += 1

		# determine neighbors 
		for i in range(len(ele)):

			for j in range(len(ele)):

				# test lower neighbor 
				if (ele[i].F[0,0] == ele[j].F[2,0] and ele[i].F[0,1] == ele[j].F[2,1]):

					ele[i].neighbor[0] = ele[j].elnum

				# test right 
				if (ele[i].F[1,0] == ele[j].F[3,0] and ele[i].F[1,1] == ele[j].F[3,1]):

					ele[i].neighbor[1] = ele[j].elnum

				# test top 
				if (ele[i].F[2,0] == ele[j].F[0,0] and ele[i].F[2,1] == ele[j].F[0,1]):

					ele[i].neighbor[2] = ele[j].elnum

				# test left 
				if (ele[i].F[3,0] == ele[j].F[1,0] and ele[i].F[3,1] == ele[j].F[1,1]):

					ele[i].neighbor[3] = ele[j].elnum

			print(ele[i].neighbor)

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

		self.J = np.zeros((2,2)) # jacobian 

	def jacobian(self):

		# how to compute jacobian 


	# def genStiff(self, a, b, c, q):

	# 	func = lambda xi, eta: 


mesh = Mesh(2, 2, 1, 1)