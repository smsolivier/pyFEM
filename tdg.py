#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from genBasis import genBasis 

from scipy.integrate import quadrature as quad 

def TDG(Nx, Nt, xb, tb, p, f0x, f0t, a, b, c, q):

	alpha = 1

	h = xb/Nx # element width 

	dt = tb/Nt # time step 

	f = np.zeros(Nx)

	x = np.linspace(0, xb, Nx+1) 
	t = np.linspace(0, tb, Nt+1)

	fprev_new = np.ones((Nx, p))*f0t # store previous time step for each element and node 

	for l in range(1, Nt): # loop over time steps 

		fprev = np.copy(fprev_new)

		for i in range(Nx): # loop over elements 

			# generate basis functions 
			B, dB = genBasis(p)

			xglob = np.linspace(x[i], x[i+1], p) # global x values, evenly spaced 

			# map xi --> x 
			X = lambda xi: sum([xglob[j]*B[j](xi) for j in range(p)])

			# jacobian 
			J = lambda xi: sum([xglob[j]*dB[j](xi) for j in range(p)])

			# store local system 
			A = np.zeros((p,p))
			rhs = np.zeros(p) 

			for j in range(p): # loop over nodes 

				for k in range(p): # loop over nodal contributions 

					# compute mass matrix 
					func = lambda xi: B[j](xi) * B[k](xi) * J(xi) 
					Mjk = quad(func, -1, 1)[0] 

					# compute stiffness matrix 
					func = lambda xi: dB[j](xi) * B[k](xi) 
					Sjk = quad(func, -1, 1)[0] 

					A[j,k] = a/dt*Mjk + alpha*(-b*Sjk + c*Mjk + b*B[j](1)*B[k](1))

					# previous time step info 
					rhs[j] += fprev[i,k]*(a/dt*Mjk - 
						(1-alpha)*(-b*Sjk + c*Mjk + b*B[j](1)*B[k](1)))

					# source 
					rhs[j] += (alpha*q(xglob[j], t[l]) + (1-alpha)*q(xglob[j], t[l-1]))*Mjk

					# upwinding (current and previous time step)
					if (i == 0):

						fnodes = np.ones(p)*f0x # boundary condition 

					rhs[j] += b*B[j](-1)*B[k](-1)*(alpha*fnodes[-1] + (1-alpha)*fprev[i-1,-1])

			fnodes = np.linalg.solve(A, rhs) # solve local system 

			f_func = lambda xi: sum([B[j](xi)*fnodes[j] for j in range(p)]) # interpolated result 

			f[i] = f_func(0) # cell centered value 

			fprev_new[i,:] = fnodes

		xc = np.linspace(h/2, xb-h/2, Nx)
		file = open('data/'+str(l)+'.curve', 'w')
		file.write('# curve\n')
		for i in range(Nx):

			file.write('{} {}\n'.format(xc[i], f[i]))

		file.close()

	return np.linspace(h/2, xb-h/2, Nx), f 

Nx = 30
Nt = 20
xb = 1 
tb = 2
p = 4
f0x = 0
f0t = 0
a = 0
b = 1 
c = 1 
q = lambda x, t: a/tb*np.sin(np.pi*x/xb) + b*np.pi/xb/tb*np.cos(np.pi*x/xb) + t/tb*np.sin(np.pi/xb)

f_mms = lambda x, t: t/tb*np.sin(np.pi*x/xb)

x, f = TDG(Nx, Nt, xb, tb, p, f0x, f0t, a, b, c, q)

plt.plot(x, f)

xt = np.linspace(0, xb, 100)
plt.plot(xt, f_mms(xt, tb))
plt.show()

# x = np.linspace(0, xb, 100)
# plt.plot(x, np.exp(tb)*np.exp(-x))
# plt.show()