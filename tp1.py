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
    B = A@A
    i = 2
    while (i < k) :
        B = B@A
        i = i+1
    return B;

def cociente_rayleigh(x,A):
    num = x @ A @ x
    den = x @ x
    return num/den    

def vector_random(long):
    v = np.random.rand(long)
    return v

A = np.array([[-2.,0,0],[0,-5.,6.],[0,-3.,4.]])
B2 = producto_matriz_k_veces(A, 50)
v1 = vector_random(B.shape[0])

a1 = cociente_rayleigh(B2@v1 , B2)
print(a1)


#print( np.linalg.eigvals(B))


# A = matriz a la que le va a calcular los autovalores




#%% EJERCICIO 2