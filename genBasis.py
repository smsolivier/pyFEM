#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

def genBasis(N):

	x = np.linspace(-1, 1, N)

	coef = np.zeros((N, N))

	for k in range(N):

		A = np.zeros((N, N))

		for i in range(N):

			for j in range(N):

				A[i,j] = x[i]**j

		b = np.zeros(N)
		b[k] = 1 

		coef[k,:] = np.linalg.solve(A, b)

	B = [] 
	dB = [] 

	x2 = np.linspace(-1, 1)
	for i in range(N):

		B.append(np.poly1d(coef[i,::-1]))

		dB.append(B[i].deriv())

	return B, dB

def genBasis2D():

	pts = np.array([[-1,-1], [1,-1], [1,1], [-1,1]])

	coef = np.zeros((4,4))

	for i in range(4):

		A = np.zeros((4,4))

		for j in range(4):

			A[j,:] = np.array([1, pts[j,0], pts[j,1], pts[j,0]*pts[j,1]])

		b = np.zeros(4)
		b[i] = 1 

		coef[i,:] = np.linalg.solve(A, b)

	B = [] 
	dBxi = [] 
	dBeta = [] 

	for i in range(4):

		B.append(lambda x, y: coef[i,0] + coef[i,1]*x + coef[i,2]*y + coef[i,3]*x*y)

		dBxi.append(lambda x, y: coef[i,1] + coef[i,3]*y)

		dBeta.append(lambda x, y: coef[i,2] + coef[i,3]*x)

	return B, dBxi, dBeta 

if __name__ == '__main__': 

	# N = 4
	# B, dB = genBasis(N)

	# xloc = np.linspace(-1, 1, N)

	# x = np.linspace(-1, 1, 100)
	# for i in range(N):

	# 	plt.plot(x, B[i](x))

	# 	plt.axvline(xloc[i], color='k', alpha=.5)

	# plt.show()

	B, dBxi, dBeta = genBasis2D()