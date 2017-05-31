#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from genBasis import genBasis 

from scipy.integrate import quadrature as quad 
from scipy.interpolate import interp1d 

def DG(N, p, xb, f0, a, b, q):

	h = xb/N  

	f = np.zeros(N)
	fe = np.zeros(N+1) 
	fe[0] = f0 

	x = np.linspace(0, xb, N+1)

	for i in range(N):

		# generate basis functions 
		B, dB = genBasis(p)

		xglob = np.linspace(x[i], x[i+1], p) # global x values 

		# function to map xi --> global x  
		X = lambda xi: sum([xglob[i]*B[i](xi) for i in range(p)])

		# compute Jacobian = sum x_glob,i dB_i/dxi 
		J = lambda xi: sum([xglob[i]*dB[i](xi) for i in range(p)]) 

		A = np.zeros((p,p)) 
		rhs = np.zeros(p) 

		for j in range(p):

			for k in range(p): 

				func = lambda xi: -a*B[k](xi)*dB[j](xi) + b*B[j](xi)*B[k](xi) * J(xi) 
				A[j,k] = quad(func, -1, 1)[0] 

				func = lambda xi: B[j](xi)*B[k](xi)*q(xglob[j]) * J(xi)
				rhs[j] += quad(func, -1, 1)[0] 

		A[-1,-1] += a*B[-1](1) 

		if (i != 0):

			rhs[0] += a*B[0](-1)*fnodes[-1]  

		else:

			rhs[0] += a*B[0](-1)*f0

		fnodes = np.linalg.solve(A, rhs) # solve local system 

		f_func = lambda xi: sum([B[i](xi)*fnodes[i] for i in range(p)]) # interpolated result 

		f[i] = f_func(0) # cell centered value 
		fe[i+1] = fnodes[-1] # edge value 

		# xres = np.linspace(-1, 1, 100)
		# plt.plot(X(xres), f_func(xres))
		# plt.plot(xglob, fnodes, 'o')

	# return np.linspace(h/2, xb-h/2, N), f
	return x, fe 

def order(p):

	Nrun = 4

	N = np.array([int(n) for n in np.logspace(1, 2, Nrun)])
	err = np.zeros(Nrun)

	for i in range(Nrun):

		x, f = DG(N[i], p, xb, f0, a, b, q)

		interp = interp1d(x, f)

		err[i] = np.max(np.fabs(f - f_mms(x)))

	fit = np.polyfit(np.log(xb/N), np.log(err), 1)
	print(fit[0])

	plt.loglog(xb/N, err, '-o', label=str(p))

a = 1 
b = 1 

xb = 1 

q = lambda x: a*np.pi/xb*np.cos(np.pi*x/xb) + b*np.sin(np.pi*x/xb)
# q = lambda x: 0 

p = 2

f0 = 0

N = 4

# x, f = DG(N, p, 2*xb, f0, a, b, q)

# xmms = np.linspace(0, 2*xb, 100)
f_mms = lambda x: np.sin(np.pi*x/xb)

# plt.plot(x, f, '-o')
# plt.plot(xmms, f_mms(xmms))

# plt.show()

p = np.arange(2, 6)

for i in range(len(p)):

	order(p[i])

plt.legend(loc='best')
plt.show()


