""" Autor V M Krushnarao Kotteda (Murali)                                                                                                                                              
date Oct 31 2020 """

#https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html
import sys
import scipy.sparse as sp
import scipy.sparse.linalg
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy import linalg
from scipy.sparse import identity
import time


class gmres_counter(object):
    def __init__(self, disp=True):
        self._disp = disp
        self.niter = 0
    def __call__(self, rk=None):
        self.niter += 1
        if self._disp:
            print('iter %3i\trk = %s' % (self.niter, str(rk)))

#A = np.array([[1, 2, 0, 0], [0, 3, 4, 0], [0, 0, 5, 6], [7, 0, 8, 9]])
#A = sp.csr_matrix(A)
N = 50 
#A = np.array([    [5, 2, 1, 1],     [2, 6, 2, 1],     [1, 2, 7, 1],     [1, 1, 2, 8] ])
#A = sp.diag([[4, 2, 1, 1],     [2, 5, 2, 1],     [1, 2, 4, 1],     [1, 1, 2, 4] ], shape=[N, N], format='csc')
#b = np.array([29, 31, 26, 19]) 
initial_guess = np.ones(4)
# 4x1  + 2x2 + x3 + x4 = 29 
# 2x1  + 5x2 + 2x3 + x4 = 31 
# x1   + 2x2 + 4x3 + x4 = 26 
# x1   + x2  + 2x3 + 4x4 = 19 

A = sp.diags([1, -2, 1], [1, 0, -1], shape=[N, N], format='csc')
print(A) 
#A = sp.rand(N, N, density=0.5)
b = np.ones(N) 
D = -2 * identity(N) # np.zeros([N,N]) 
#print(D) 
invD = scipy.sparse.linalg.inv(D)
#print(invD) 

# sparse solve 
#x = sp.linalg.spsolve(A, b)
#print(np.linalg.norm(A * x - b))

# dense direct solver 
x = np.linalg.solve(A.todense(), b)
print(x) 

#LU factorization Ax = b -> LUx = b 
lu = sp.linalg.splu(A) # sp.sparse.splu  or sp.sparse.spilu 
x = lu.solve(b)
#print(lu.L) 
#print(lu.U) 
#print(x) 

counter = gmres_counter()
print(" CG solver")
t0 = time.clock()
x, info = sp.linalg.cg(A, b,callback=counter)
t1 = time.clock()
print(x)
print("wall clock time",t1-t0)

print("GMRES solver")
t2 = time.clock()
x, info = sp.linalg.bicgstab(A, b,maxiter=1e6, tol= 1e-10, callback=counter)
t3 = time.clock()
print(x)
print("CG wall clock time",t1-t0)
print("GMRES wall clock time",t3-t2)

sys.exit("stop") 
print(" Preconditioned CG solver") 
x, info = sp.linalg.cg(A, b,M=invD,callback=counter)
print(x)

sys.exit("stop") 

print("BiCG solver") 
x, info = sp.linalg.bicg(A, b,maxiter=10, callback=counter)
print(x)

print("BiCGStab solver") 
x, info = sp.linalg.bicgstab(A, b,maxiter=10, callback=counter)
print(x)

print("GMRES solver") 
x, info = sp.linalg.gmres(A, b,maxiter=10, callback=counter)
print(x)

sys.exit("stop") 

print("LGMRES solver") 
x, info = sp.linalg.lgmres(A, b,atol=1e-10,maxiter=10, callback=counter)
print(x)

print("MINRES solver") 
x, info = sp.linalg.minres(A, b,maxiter=10, callback=counter)
print(x)

print("QMR solver") 
x, info = sp.linalg.qmr(A, b,maxiter=10, callback=counter)
print(x)
