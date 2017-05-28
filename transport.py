#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from fem import * 

from scipy.interpolate import interp1d 

N = 10 
n = 8
order = 2
sigmat = 1 
sigmas = .1 

xb = 1 

fplus = 1 
fminus = 0 

Q = 1

NN = (order-1)*N + 1 

phi = np.zeros(NN)

mu, w = np.polynomial.legendre.leggauss(n)

tol = 1e-3 

x = np.linspace(0, xb, NN)

it = 0 
while(True):

	phi_old = np.copy(phi)

	phi = np.zeros(NN)

	for i in range(n):

		if (mu[i] > 0):

			BCL = 1
			f0 = 0

		else:

			BCL = 0 
			f0 = 0 

		# phi is vector, Q is supposed to be constant 
		phi_func = interp1d(x, phi_old)
		x, psi = FEM(N, xb, mu[i], sigmat, lambda x: sigmas/2*phi_func(x) + Q, f0, BCL, order)

		phi += w[i]*psi[:] 

	norm = np.linalg.norm(phi - phi_old, 2)/np.linalg.norm(phi, 2)

	if (norm < tol): 

		break 

	it += 1 

	print(it, end='\r')

print('Number of Iterations =', it)

plt.plot(x, phi, '-o')
plt.show()

