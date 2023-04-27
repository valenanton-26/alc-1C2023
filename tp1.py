# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matplotlib.pyplot as plt
import numpy as np
import math

#%% EJERCICIO 1
# Desarrollar un programa que dada una una matriz A ∈ Rn×n y un entero positivo k, realice k iteraciones del metodo de la 
# potencia con un vector aleatorio inicial v ∈ Rn. El programa debe devolver un vector a ∈ Rk, donde ai sea la 
# aproximacion al autovalor obtenida en el paso i.


def producto_matriz_k_veces(A, k): #se asume k>0
    B = A
    i = 1
    while (i < k):  #B = A^i al finalizar cada iteración del ciclo. Por eso i<k
        B = B@A
        i = i+1
    return B;

def cociente_rayleigh(x,A):
    num = x @ A @ x
    den = x @ x
    return (num/den)    

def autovector_generado(matriz, vect):
    prod = matriz@vect
    # || A^k * v1 ||2
    norma = np.linalg.norm(prod, 2)
    
    return prod/norma


def generar_autovalor(M, v, k):
    # el resultado de multiplicar k veces a la matriz A
    B2 = producto_matriz_k_veces(M, k)
    # calculo del autovector
    autovector1 = autovector_generado(B2, v)
    
    a1 = cociente_rayleigh(autovector1 , M)
    return a1;

def generar_vector(M, v, k):
    a = []
    i = 0 #cuenta iteraciones del ciclo
    while i < k:
        autovalor = generar_autovalor(M, v, i+1)
        a = np.append(a, autovalor)
        print(a)
        i += 1
    return a

# A = matriz a la que le va a calcular los autovalores
A = np.array([[-2.,0,0],[0,-5.,6.],[0,-3.,4.]])

# vector random de la longitud de los vectores de la matriz inicial
v1 = np.random.rand(A.shape[0])

print(v1)
print(generar_vector(A,v1,10))

# para verificar
print( np.linalg.eigvals(A))

#EJERCICIO 2

def generarMatrizTipoA(n):
    A = np.random.randn(n,n)
    # genera una matriz de elementos reales aleatorios, de forma nxn
    return A

def generarMatrizTipoB(n):
    B = np.random.randn(n,n)
    i  = 0
    while i < (n-1):
        j = i+1
        while j < n:
            B[i][j] = B[j][i]
            j +=1
        i +=1
    return B

def generarMatrizTipoC(M):
    C = M.copy()
    i = 0
    while i < M.shape[0]:
        C[i][i] = (C[i][i]) + 100
        i += 1
    return C

def generarMatrizTipoD(M):
    C = M.copy()
    i = 0
    while i < M.shape[0]:
        C[i][i] = (C[i][i]) + 1000
        i += 1
    return C

def generarMatrices(n):
    A = generarMatrizTipoA(n)
    B = generarMatrizTipoB(n)
    C = generarMatrizTipoC(B)
    D = generarMatrizTipoD(B)
    return A, B, C, D

# Item A

