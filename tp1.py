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

def producto_matriz_k_veces(A, k):
    B = A
    i = 1
    while (i < k) :
        B = B@A
        i = i+1
    return B;

def cociente_rayleigh(x,A):
    num = x @ A @ x
    den = x @ x
    return np.round(num/den)    

def vector_random_long_k(long):
    v = np.random.rand(long)
    return v

def autovector_generado(matriz, vect):
    prod = matriz@vect
    # || A^k * v1 ||2
    norma = np.linalg.norm(prod, 2)
    
    return prod/norma
    

# A = matriz a la que le va a calcular los autovalores
A = np.array([[-2.,0,0],[0,-5.,6.],[0,-3.,4.]])

# el resultado de multiplicar k veces a la matriz A
B2 = producto_matriz_k_veces(A, 50)

# vector random de la longitud de los vectores de la matriz inicial
v1 = vector_random_long_k(B2.shape[0])

# calculo del autovector
autovector1 = autovector_generado(B2, v1)

a1 = cociente_rayleigh(autovector1 , A)
print(a1)


print( np.linalg.eigvals(A))


#%% EJERCICIO 2


#%% EJERCICIO 3

