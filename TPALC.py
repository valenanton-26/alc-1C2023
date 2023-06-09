# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

encabezado = np.arange(0, 785, 1)
datos = pd.read_csv("~/Downloads/mnist_train.csv", names=encabezado)
df_testeo = datos = pd.read_csv("~/Downloads/mnist_test.csv", names=encabezado)

"""
EJERCICIO 1
"""
#ITEM A

#Funcion que dado un array correspondiente al gráfico de un dígito, devuelve su gráfico
def graficar(v):
    titulo = v[0]
    figura = v[1:].reshape(28, 28)
    
    plt.imshow(figura, cmap='gray')
    plt.title(titulo)
    plt.show()
    plt.close()

def imagen (n, datos):
    fila = datos.iloc[n:n+1]
    imagen = fila.to_numpy()
    imagen = imagen[0] #to_numpy devuelve una matriz cuyo primer elemento es el array con los datos
    graficar(imagen)

#ITEM B

#Funcion que, dado un dígito y una base de datos devuelve un nuevo df con las filas correspondientes a ese digito
def df_digito (n, datos):
    nombre_columna = datos.columns[0]
    df = datos[datos[nombre_columna] == n]
    return df

#Dada una base de datos y un dígito, la función devuelve la cantidad de filas que corresponden a ese dígito
def cantidad(n, datos):
    df = df_digito(n, datos)
    cantidad = df.shape[0]
    return cantidad

#ITEM C

#Función que, dado un dígito y un conjunto de datos, devuelve un array con la imagen promedio de cada dígito
def promedio(n, datos):
    df = df_digito(n, datos)
    p = []
    for i in range(0, 785, 1):
        promedio_columna = df[i].mean()
        p.append(promedio_columna)
    
    p = np.array(p)
    return p

datos_reducidos = datos.head(2000)
for i in range(0, 10, 1):
    graficar(promedio(i, datos_reducidos))
    
"""
EJERCICIO 2
"""

#ITEM A

#Definimos una matriz con los promedios calculados para cada uno de los dígios. 

#Lo dejo como matriz porque es más cómodo para el punto de las predicciones, pero si es por
#la prolijidad podemos hacer un df y después convertirlo a array

matriz_promedios = np.array([promedio(0, datos_reducidos)])
for i in range(1, 10, 1):
    matriz_promedios = np.vstack([matriz_promedios, promedio(i, datos_reducidos)])


#Defino una función que, dados los promedios (matriz), un conjunto de datos de testeo
#y una fila determinada, devuelve la predicción del dígito, en base a las distancias euclídeas
def prediccion(promedios, test, n):
    pred = 0
    fila = test.iloc[n:n+1]
    fila = fila.to_numpy()
    fila = fila[0]
    dist_min = np.linalg.norm(fila - promedios[0])
    k = 1
    while k<10:
        distancia = np.linalg.norm(fila - promedios[k])
        if distancia < dist_min:
            dist_min = distancia
            pred = k
        k+=1
    return pred


def lista_predicciones(promedios, test):
    predicciones = []
    l = test.shape[0]
    for i in range (0, l, 1):
        pred = prediccion(promedios, test, i)
        predicciones.append(pred)
     
    predicciones = np.array(predicciones)
    return predicciones


test_reducido = df_testeo.head(200)
predicciones = lista_predicciones(matriz_promedios, test_reducido)


#ITEM B

def precision(test, prediccion):
    aciertos = 0
    l = len(prediccion)
    for i in range(0, l, 1):
        p = prediccion[i]
        rta = test[0][i]
        if p == rta:
            aciertos+=1
    
    prec = aciertos/l
    return prec


#ITEM C

#Defino una función que, dada una lista de predicciones y un conjunto de datos de testeo
#Devuelve un lista con los números de las filas en las cuales falló la predicción

def errores(test, prediccion):
    e = []
    l = len(prediccion)
    for i in range(0, l, 1):
        p = prediccion[i]
        rta = test[0][i]
        if p != rta:
            e.append(i)
            
    return e

e = errores(test_reducido, predicciones)

imagen(46, test_reducido)
p = prediccion(matriz_promedios, test_reducido, 46)
print(p)
