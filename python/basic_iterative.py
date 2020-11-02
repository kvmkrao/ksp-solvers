""" Autor V M Krushnarao Kotteda (Murali) 
date Oct 31 2020 """

import numpy as np
import numpy.linalg as nl

def Jacobi_expand(A, b, n):
    n = len(b) 
    X = np.zeros(n)
    Y = np.zeros(n)
    D = np.zeros(n)
    X1 = np.zeros([n, 1])
    b1 = np.zeros([n, 1])
    for i in range(0, n):
        b1[i][0] = b[i]
    k = 0
    while (True):
        for i in range(0, n):
            tmp = 0
            for j in range(0, n):
                if j != i:
                    tmp += A[i][j] * X[j]
            Y[i] = (b[i] - tmp) / A[i][i]
        for i in range(0, n):
            X1[i][0] = Y[i]
        D = np.dot(A, X1)
        D = D - b1
        k = k + 1
        if k >= 100:
            break
        if  (nl.norm(D) / nl.norm(b) < 1e-6) & (nl.norm(D) / nl.norm(b) > -(1e-6)):
            break
        X = Y.copy()
    return Y


def jacobi(A, b, x_init, epsilon=1e-6, max_iterations=100):
    D = np.diag(np.diag(A))
    LU = A - D
    x = x_init
    for i in range(max_iterations):
        D_inv = np.diag(1 / np.diag(D))
        x_new = np.dot(D_inv, b - np.dot(LU, x))
        residual = np.linalg.norm(x_new - x)
        print('i, Residual %s' %[i,residual])
        if np.linalg.norm(x_new - x) < epsilon:
            return x_new
        x = x_new
    return x

def GaussSeidel(A, b, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    D = np.zeros(n)
    X1 = np.zeros([n, 1])
    b1 = np.zeros([n, 1])
    for i in range(0, n):
        b1[i][0] = b[i]
    k = 0
    while (True):
        for i in range(0, n):
            tmp = 0
            for j in range(0, n):
                if j < i:
                    tmp += A[i][j] * Y[j]
                if j > i:
                    tmp += A[i][j] * X[j]
            Y[i] = (b[i] - tmp) / A[i][i]
        for i in range(0, n):
            X1[i][0] = Y[i]
        D = np.dot(A, X1)
        D = D - b1
        k = k + 1
        if k >= 50:
            break
        if  (nl.norm(D) / nl.norm(b) < 1e-6) & (nl.norm(D) / nl.norm(b) > -(1e-6)):
            break
        X = Y.copy()
    return Y


def sor_solver(A, b, omega, initial_guess, convergence_criteria):
  """
  This is an implementation of the pseduo-code provided in the Wikipedia article.
  Inputs:
    A: nxn numpy matrix
    b: n dimensional numpy vector
    omega: relaxation factor
    initial_guess: An initial solution guess for the solver to start with
  Returns:
    phi: solution vector of dimension n
  """
  phi = initial_guess[:]
  residual = np.linalg.norm(np.matmul(A, phi) - b) #Initial residual
  ic = 0 
  while residual > convergence_criteria:
    ic = ic + 1 
    for i in range(A.shape[0]):
      sigma = 0
      for j in range(A.shape[1]):
        if j != i:
          sigma += A[i][j] * phi[j]
      phi[i] = (1 - omega) * phi[i] + (omega / A[i][i]) * (b[i] - sigma)
    residual = np.linalg.norm(np.matmul(A, phi) - b)
#    print('Residual: {0:10.6g}'.format(residual))
    print('ic, Residual %s' %[ic,residual])
  return phi

def SOR(A, b, n, w):
    n = len(b) 
    X = np.zeros(n)
    Y = np.zeros(n)
    D = np.zeros(n)
    X1 = np.zeros([n, 1])
    b1 = np.zeros([n, 1])
    for i in range(0, n):
        b1[i][0] = b[i]
    k = 0
    while (True):
        for i in range(0, n):
            tmp1 = 0
            tmp2 = 0
            for j in range(0, n):
                if j < i:
                    tmp1 += A[i][j] * Y[j]
                if j > i:
                    tmp2 += A[i][j] * X[j]
            Y[i] = X[i] + (w * (b[i] - tmp1 - tmp2)) / A[i][i]
        for i in range(0, n):
            X1[i][0] = Y[i]
        D = np.dot(A, X1)
        D = D - b1
        k = k + 1
        if k >= 50:
            break
        if  (nl.norm(D) / nl.norm(b) < 1e-6) & (nl.norm(D) / nl.norm(b) > -(1e-6)):
            break
        X = Y.copy()


#An example case that mirrors the one in the Wikipedia article
residual_convergence = 1e-8
omega = 0.5 #Relaxation factor

#A = np.array([ [4, -1, -6, 0], [-5, -4, 10, 8], [0,9,4,-2], [1,0,-7,5]]) 
#b = np.array([2,21,-12,-6]) 
A = np.array([    [5, 2, 1, 1],     [2, 6, 2, 1],     [1, 2, 7, 1],     [1, 1, 2, 8] ])
#A = np.array([    [3, 2, 1, 1],     [2, 4, 2, 1],     [1, 2, 3, 1],     [1, 1, 2, 3] ])
b = np.array([29, 31, 26, 19]) 

#A = np.zeros([100, 100]);
#for i in range(0, 100):
#    A[i][i] = 2
#    if i == 0:
#        A[i][i + 1] = -1
#    elif i == 99:
#        A[i][i - 1] = -1
#    else:
#        A[i][i - 1] = A[i][i + 1] = -1

initial_guess = np.ones(4)
# 4x1  + 2x2 + x3 + x4 = 29 
# 2x1  + 5x2 + 2x3 + x4 = 31 
# x1   + 2x2 + 4x3 + x4 = 26 
# x1   + x2  + 2x3 + 4x4 = 19 
#n = 500 
x = sor_solver(A, b, omega, initial_guess, 1e-6)
print("SOR")
print(x)
x = jacobi(A, b, initial_guess)
residual = np.linalg.norm(np.matmul(A, x) - b)
print('Residual: {0:10.6g}'.format(residual))
print(x)
print(np.matmul(A, x) - b)
#w = 0.5
#x=sor(A,b,n,w) 
#residual = np.linalg.norm(np.matmul(A, x) - b)
#print('Residual: {0:10.6g}'.format(residual))
#print(x)

n=len(b) 
result1 = Jacobi_expand(A, b, n)
print("Jacobi")
print(result1)
result2 = GaussSeidel(A, b, n)
print("Gauss seidel")
print(result2)
#result3 = SOR(A, b, n, 0.5)
#print("SOR")
#print(result3)

