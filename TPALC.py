# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pandas as pd
import csv
import matplotlib.pyplot as plt
import numpy as np

encabezado = np.arange(0, 785, 1)
datos = pd.read_csv("~/Descargas/mnist_train.csv", names=encabezado)

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
    datos = fila.to_numpy()
    datos = datos[0] #to_numpy devuelve una matriz cuyo primer elemento es el array con los datos
    graficar(datos)

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

def promedio(n, datos):
    df = df_digito(n, datos)
    p = []
    for i in range(0, 785, 1):
        promedio_columna = df[i].mean()
        p.append(promedio_columna)
    
    p = np.array(p)
    graficar(p)


datos_reducidos = datos.head(2000)
for i in range(1, 10, 1):
    promedio(i, datos_reducidos)
