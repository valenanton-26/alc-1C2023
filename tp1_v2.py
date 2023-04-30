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
    i = 1 
    while i<=k:
        matriz = producto_matriz_k_veces(A, i)
        Av = matriz@res
        res = Av/np.linalg.norm(res, 2)
        i+=1
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

Matrices = generarMatrices(100)
v = np.random.rand(Matrices[0].shape[0])
#Para graficar, creo una lista con todos los números de cada una de las iteraciones
iteraciones = list(range(0, 101)) 

A = Matrices[0]
aproximaciones_A = vector_autovalores(A, v, 100)
plt.plot(iteraciones, aproximaciones_A, color = 'b')
plt.title('Matriz tipo A')
plt.xlabel('Número de iteraciones')
plt.ylabel('Aprox. del autovalor')
plt.show()
plt.close()

B = Matrices[1]
aproximaciones_B = vector_autovalores(B, v, 100)
plt.plot(iteraciones, aproximaciones_B, color = 'b')
plt.title('Matriz tipo B')
plt.xlabel('Número de iteraciones')
plt.ylabel('Aprox. del autovalor')
plt.show()
plt.close()

C = Matrices[2]
aproximaciones_C = vector_autovalores(C, v, 100)
plt.plot(iteraciones, aproximaciones_C, color = 'b')
plt.title('Matriz tipo C')
plt.xlabel('Número de iteraciones')
plt.ylabel('Aprox. del autovalor')
plt.show()
plt.close()

D = Matrices[3]
aproximaciones_D = vector_autovalores(D, v, 100)
plt.plot(iteraciones, aproximaciones_D, color = 'b')
plt.title('Matriz tipo D')
plt.xlabel('Número de iteraciones')
plt.ylabel('Aprox. del autovalor')
plt.show()
plt.close()

#EJERCICIO 3

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

#Este ejemplo devuelve 3,00..4 en vez de 3 (la función abs devuelve los números como punto flotante). 
#Preguntar si se debería redondear o dejar así
"""
prueba = np.array([[1, 2, 0], [2, 1, 0], [4, 1, 0]])
a = np.linalg.eigvals(prueba)
print(a)
v = autovalor_max(prueba)
print(v)
"""

#función que, dado un autovalor y una aproximación, calcula el valor del error como se encuentra definido en la consigna
def error(autov, aprox):
    e = abs(autov - aprox)
    return e

#función que, dado una autovalor y una lista l de aproximaciones, devuelve una lista r tal que,
#r[i] representa el error correspondiente a la aproximación l[i]
def vector_errores(autov, lista):
    res = np.array([])
    for l in lista:
        e = error(autov, l)
        res = np.append(res, e)
    return res

#Dado un vector v, devuelve una lista con el logaritmo en base 10 de cada uno de sus elementos
def log_en_base_10 (vector):
    res = []
    for elemento in vector:
        log = np.log10(elemento)
        res.append(log)
    return res
    
#ITEM A y B

#Para graficar, defino como eje x la lista con todos los números de cada una de las iteraciones realizadas
x = np.linspace(0, 100, 101)


#Caso A
errores_A = vector_errores(autovalor_max(A), aproximaciones_A) #creo el vector con sus respectivos errores
log_errores_A = log_en_base_10(errores_A) #calculo el logaritmo en base 10 de cada uno de los errores obtenidos

#Calculamos el vector formado por el módulo de autovalores de A y ordenamos sus elementos de mayor a menor
modulo_autovalores_A = valores_absolutos(np.linalg.eigvals(A))
modulo_autovalores_A = sorted(modulo_autovalores_A, reverse=True)

#Ahora defino la función
autov1_A = modulo_autovalores_A[0]
autov2_A = modulo_autovalores_A[1]
funcion_A = 2 * np.log10(autov2_A/autov1_A) * x + np.log10(errores_A[0])

fig, ax = plt.subplots()
ax.plot(x, log_errores_A, label = 'Error Matriz A', color = 'b')
ax.plot(x, funcion_A, label = 'Función', color = 'r')
ax.set_xlabel('Número de iteraciones')
ax.set_ylabel('Error')
ax.set_title('Error en matrices de tipo A')

ax.legend()

plt.show()
plt.close()

#Caso B
errores_B = vector_errores(autovalor_max(B), aproximaciones_B)
log_errores_B = log_en_base_10(errores_B)

#Calculamos el vector formado por el módulo de autovalores de B y ordenamos sus elementos de mayor a menor
modulo_autovalores_B = valores_absolutos(np.linalg.eigvals(B))
modulo_autovalores_B = sorted(modulo_autovalores_B, reverse=True)

#Ahora defino la función
autov1_B = modulo_autovalores_B[0]
autov2_B = modulo_autovalores_B[1]
funcion_B = 2 * np.log10(autov2_B/autov1_B) * x + np.log10(errores_B[0])

fig, ax = plt.subplots()
ax.plot(x, log_errores_B, label = 'Error Matriz B', color = 'b')
ax.plot(x, funcion_B, label = 'Función', color = 'r')
ax.set_xlabel('Número de iteraciones')
ax.set_ylabel('Error')
ax.set_title('Error en matrices de tipo B')

ax.legend()

plt.show()
plt.close()

#Caso C
errores_C = vector_errores(autovalor_max(C), aproximaciones_C)
log_errores_C = log_en_base_10(errores_C)

#Calculamos el vector formado por el módulo de autovalores de C y ordenamos sus elementos de mayor a menor
modulo_autovalores_C = valores_absolutos(np.linalg.eigvals(C))
modulo_autovalores_C = sorted(modulo_autovalores_C, reverse=True)

#Ahora defino la función
autov1_C = modulo_autovalores_C[0]
autov2_C = modulo_autovalores_C[1]
funcion_C = 2 * np.log10(autov2_C/autov1_C) * x + np.log10(errores_C[0])

fig, ax = plt.subplots()
ax.plot(x, log_errores_C, label = 'Error Matriz C', color = 'b')
ax.plot(x, funcion_C, label = 'Función', color = 'r')
ax.set_xlabel('Número de iteraciones')
ax.set_ylabel('Error')
ax.set_title('Error en matrices de tipo C')

ax.legend()

plt.show()
plt.close()

#Caso D
errores_D = vector_errores(autovalor_max(D), aproximaciones_D)
log_errores_D = log_en_base_10(errores_D)

#Calculamos el vector formado por el módulo de autovalores de D y ordenamos sus elementos de mayor a menor
modulo_autovalores_D = valores_absolutos(np.linalg.eigvals(D))
modulo_autovalores_D = sorted(modulo_autovalores_D, reverse=True)

#Ahora defino la función
autov1_D = modulo_autovalores_D[0]
autov2_D = modulo_autovalores_D[1]
funcion_D = 2 * np.log10(autov2_D/autov1_D) * x + np.log10(errores_D[0])

fig, ax = plt.subplots()
ax.plot(x, log_errores_D, label = 'Error Matriz D', color = 'b')
ax.plot(x, funcion_D, label = 'Función', color = 'r')
ax.set_xlabel('Número de iteraciones')
ax.set_ylabel('Error')
ax.set_title('Error en matrices de tipo D')

ax.legend()

plt.show()
plt.close()

