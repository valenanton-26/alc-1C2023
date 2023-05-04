# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 10:34:26 2023

@author: Lara
"""

import matplotlib.pyplot as plt
import numpy as np
import math

#EJERCICIO 1

def producto_matriz_k_veces(A, k): #se asume k>0
    B = A
    i = 1
    while (i < k):  #B = A^i al finalizar cada iteración del ciclo. Por eso i<k
        B = B@A
        i = i+1
    return B;

#Dada una matriz cuadrada A de nxn, un vector v cualquiera de n elementos y un entero positivo k,
#esta función realiza k iteraciones del método de la potencia con A y v
def metodo_de_la_potencia(A, v, k):
    res = v.copy()
    if k>=1:
        matriz = producto_matriz_k_veces(A, k)
        Av = matriz@res
        res = Av/np.linalg.norm(Av, 2)
    return res

#Dada una matriz cuadrada A de nxn y v vector de longitud n, esta función calcula el cociente de Rayleigh
def cociente_rayleigh(v,A):
    num = v @ A @ v
    den = v @ v
    return num/den  

#Dada una matriz cuadrada A de nxn, un vector v de longitud n y un entero positivo k,
#esta función devuelve una aproximación al autovalor de la matriz A,
#obtenida tras realizar k iteraciones del método de la potencia con el vector v
def autovalor_aproximado (A, v, k):
    autovector = metodo_de_la_potencia(A, v, k)
    autovalor = cociente_rayleigh(autovector, A)
    return autovalor

#Dada una matriz diagonalizable A, un vector v y un número entero k >= 0, 
#esta función realiza k iteraciones del método de la potencia con A y v
#y devuelve una lista donde, en cada posición i, muestra el autovalor aproximado
#obtenido en la iteración i de este método.
def vector_autovalores(A, v, k):
    res = np.array([])
    i = 0
    while i <= k:
        a = autovalor_aproximado(A, v, i)
        res = np.append(res, a)
        i += 1
    return res
        
"""
matriz = np.array([[1, 2, 0], [2, 1, 0], [4, 1, 0]])
vector = np.array([0, 1, 1])
p = metodo_de_la_potencia(matriz, vector, 0)
#print(p)
autov = autovalor_aproximado(matriz, vector, 7)
print(autov)
v = vector_autovalores(matriz, vector, 7)
print(v)
"""

#EJERCICIO 2

def generarMatrizTipoA(n):
    A = np.random.rand(n,n)
    # genera una matriz de elementos reales aleatorios, de forma nxn
    return A

def generarMatrizTipoB(n):
    B = np.random.rand(n,n)
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


Matrices = generarMatrices(100)
v = np.random.rand(Matrices[0].shape[0])

def grafico_aproximaciones (M, titulo):
    #Para graficar, creo una lista con todos los números de cada una de las iteraciones
    iteraciones = list(range(0, 101))
    
    aproximaciones_M = vector_autovalores(M, v, 100)
    plt.plot(iteraciones, aproximaciones_M, color = 'b')
    plt.title(titulo)
    plt.xlabel('Número de iteraciones')
    plt.ylabel('Aprox. del autovalor')
    plt.show()
    plt.close()

#Caso A
A = Matrices[0]
grafico_aproximaciones(A, 'Matriz tipo A')

#Caso B
B = Matrices[1]
grafico_aproximaciones(B, 'Matriz tipo B')

#Caso C
C = Matrices[2]
grafico_aproximaciones(C, 'Matriz tipo C')

#Caso D
D = Matrices[3]
grafico_aproximaciones(D, 'Matriz tipo D')


#EJERCICIO 3

#Defino una función que, dada una matriz, devuelve su autovalor de mayor módulo

#Defino una función que, dada una matriz, devuelve su autovalor de mayor módulo

#Comienzo por definir una función auxiliar que, dado un vector v, devuelve un nuevo vector r
#tal que r[i] sea el valor absoluto de v[i] para todo i.
def valores_absolutos(v):
    res = np.array([])
    for elemento in v:
        res = np.append(res, abs(elemento))
    return res

def autovalor_max (A):
    autovalores = np.linalg.eigvals(A)
    mod_autovalores = valores_absolutos(autovalores) 
    res = max(mod_autovalores)
    return res


#función que, dado un autovalor y una aproximación, calcula el valor del error como se encuentra definido en la consigna
def error(autov, aprox):
    e = abs(autov - aprox)
    return e

#función que, dado una autovalor y un vector v de aproximaciones, devuelve un vector r tal que,
#r[i] representa el error correspondiente a la aproximación l[i]
def vector_errores(autov, vector):
    res = np.array([])
    for elem in vector:
        e = error(autov, elem)
        res = np.append(res, e)
    return res

#Dado un vector v, devuelve una lista con el logaritmo en base 10 de cada uno de sus elementos
def log_en_base_10 (vector):
    res = []
    for elemento in vector:
        logElem = np.log10(elemento)
        res.append(logElem)
    return res
    
#ITEM A y B

def grafico_errores(M, titulo):
    #Para graficar, defino como eje x la lista con todos los números de cada una de las iteraciones realizadas
    x = np.linspace(0, 100, 101)
    
    #Calculo errores
    aproximaciones_M = vector_autovalores(M, v, 100)
    errores_M = vector_errores(autovalor_max(M), aproximaciones_M)
    log_errores_M = log_en_base_10(errores_M) #calculo el logaritmo en base 10 de cada uno de los errores obtenidos
    
    #Calculamos el vector formado por el módulo de autovalores de A y ordenamos sus elementos de mayor a menor
    modulo_autovalores_M = valores_absolutos(np.linalg.eigvals(M))
    modulo_autovalores_M = sorted(modulo_autovalores_M, reverse=True)
    
    #Ahora definimos la función
    autov1 = modulo_autovalores_M[0]
    autov2 = modulo_autovalores_M[1]
    f = 2 * np.log10(autov2/autov1) * x + np.log10(errores_M[0])
    
    #Grafico
    fig, ax = plt.subplots()
    ax.plot(x, log_errores_M, label = 'Error Matriz', color = 'b')
    ax.plot(x, f, label = 'Función', color = 'r')
    ax.set_xlabel('Número de iteraciones')
    ax.set_ylabel('Error')
    ax.set_title(titulo)
    
    ax.legend()
    
    plt.show()
    plt.close()

#Caso A
grafico_errores(A, 'Error en matrices de tipo A')

#Caso B
grafico_errores(B, 'Error en matrices de tipo B')

#Caso A
grafico_errores(C, 'Error en matrices de tipo C')

#Caso D
grafico_errores(D, 'Error en matrices de tipo D')

