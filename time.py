#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

from genBasis import genBasis

from scipy.integrate import quadrature as quad

class Element:

        def __init__(self, pts, p):

                self.p = p

                self.xglob = np.linspace(pts[0], pts[1], p) # global x values, evenly spaced

                # generate basis functions
                self.B, self.dB = genBasis(p)

                # map xi --> x
                self.X = lambda xi: sum([self.xglob[i]*self.B[i](xi) for i in range(p)])

                # jacobian
                self.J = lambda xi: sum([self.xglob[i]*self.dB[i](xi) for i in range(p)])

                # store local system
                self.A = np.zeros((p,p))
                self.rhs = np.zeros(p)

                # store solution on each node
                self.f = np.zeros(p)

                # store nodes from previous time step
                self.f_prev = np.zeros(p)

                # generate mass, stiffness matrix
                self.M = np.zeros((p,p))
                self.S = np.zeros((p,p))

                for i in range(p):

                        for j in range(p):

                                self.M[i,j] = self.mass(i,j)
                                self.S[i,j] = self.stiff(i,j)

        def mass(self, i, j):

                func = lambda xi: self.B[i](xi)*self.B[j](xi)*self.J(xi)
                integral = quad(func, -1, 1)[0]

                return integral

        def stiff(self, i, j):

                func = lambda xi: self.dB[i](xi)*self.B[j](xi)
                integral = quad(func, -1, 1)[0]

                return integral

        def solve(self):

                self.f_prev = np.copy(self.f)

                self.f = np.linalg.solve(self.A, self.rhs)

                self.f_func = lambda xi: sum([self.f[i] * self.B[i](xi) for i in range(self.p)])

                hc = 2/self.p

                xc = np.linspace(-1+hc/2, 1-hc/2, self.p)

                # reset
                self.A = np.zeros((self.p, self.p))
                self.rhs = np.zeros(self.p)

                return self.X(xc), self.f_func(xc) # return cell center

class Mesh:

        def __init__(self, Nx, xb, p):

                self.Nx = Nx
                self.xb = xb
                self.p = p

                self.x = np.linspace(0, xb, Nx+1)

                self.h = xb/Nx # element width

                # create elements
                self.el = [] # store elements
                for i in range(Nx):

                        self.el.append(Element([self.x[i], self.x[i+1]], p))

class Solve:

        def __init__(self, mesh, Nt, tb, a, b, c, q, f0, alpha=1):

                self.mesh = mesh
                self.Nt = Nt
                self.tb = tb
                self.a = a
                self.b = b
                self.c = c
                self.q = q
                self.f0 = f0
                self.alpha = alpha

                self.dt = tb/Nt

                self.t = np.linspace(0, tb, Nt)

                self.fcount = 0

        def genLocal(self, el, upwind, upwind_prev, t, tprev):

                for i in range(el.p):

                        for j in range(el.p):

                                x = el.xglob[j] # current x location of node

                                # time
                                el.A[i,j] = self.a(x)/self.dt*el.M[i,j]

                                # space part
                                el.A[i,j] += self.alpha*(-self.b(x)*el.S[i,j] + self.c(x)*el.M[i,j])

                                el.rhs[i] += el.f[j]*self.a(x)/self.dt*el.M[i,j]

                                el.rhs[i] -= el.f[j]*(1-self.alpha)*(\
                                        -self.b(x)*el.S[i,j] + self.c(x)*el.M[i,j])

                                el.rhs[i] += self.alpha*el.M[i,j]*self.q(x,t) + \
                                        (1-self.alpha)*el.M[i,j]*self.q(x,tprev)

                # apply f(1) = f_p
                el.A[-1,-1] += self.alpha*self.b(el.X(1))
                el.rhs[-1] -= (1 - self.alpha) * el.f[-1] * self.b(el.X(1))

                # apply upwinding on left boundary
                el.rhs[0] += self.b(el.X(-1))*(self.alpha*upwind + (1-self.alpha)*upwind_prev)

        def solveSpace(self, t, tprev):

                xout = np.array([])
                fout = np.array([])

                for i in range(self.mesh.Nx):

                        el = self.mesh.el[i]

                        # generate local system
                        if (i != 0):

                                self.genLocal(el, self.mesh.el[i-1].f[-1],
                                        self.mesh.el[i-1].f_prev[-1], t, tprev)

                        else:

                                self.genLocal(el, self.f0, self.f0, t, tprev)

                        x, f = el.solve()

                        xout = np.append(xout, x)
                        fout = np.append(fout, f)

                self.writeCurve(xout, fout)

        def solveTime(self):

                for i in range(1, self.Nt):

                        self.solveSpace(self.t[i], self.t[i-1])

        def writeCurve(self, x, f):

                file = open('data/' + str(self.fcount) + '.curve', 'w')
                file.write('# curve\n')

                for i in range(len(f)):

                        file.write('{} {}\n'.format(x[i], f[i]))

                file.close()

                self.fcount += 1

Nx = 10
Nt = 40
xb = 1
tb = .01
p = 4
f0x = 0
a = lambda x: 1
b = lambda x: 1
c = lambda x: 1
alpha = .5
q = lambda x, t: a(x)*np.pi/tb*np.cos(np.pi*t/tb)*np.sin(np.pi*x/xb) \
        + b(x)*np.pi/xb*np.sin(np.pi*t/tb)*np.cos(np.pi*x/xb) \
        + c(x)*np.sin(np.pi*t/tb)*np.sin(np.pi*x/xb)

f_mms = lambda x, t: t/tb*np.sin(np.pi*x/xb)

mesh = Mesh(Nx, xb, p)
sol = Solve(mesh, Nt, tb, a, b, c, q, f0x, alpha)

sol.solveTime()