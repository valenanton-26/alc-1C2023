# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

encabezado = np.arange(0, 785, 1)
datos = pd.read_csv("~/Descargas/mnist_train.csv", names=encabezado)
df_testeo = pd.read_csv("~/Descargas/mnist_test.csv", names=encabezado)

"""
EJERCICIO 1
"""
#ITEM A

#Funcion que, dado un df y un n, devuelve la fila n del df como un array 
def fila_df(df, n):
    fila = df.iloc[n:n+1]
    fila = fila.to_numpy()
    fila = fila[0] #to_numpy devuelve una matriz cuyo primer elemento es el array con los datos
    
    return fila

#Funcion que dado un array correspondiente al gráfico de un dígito, devuelve su gráfico
def graficar(v):
    titulo = v[0]
    figura = v[1:].reshape(28, 28)
    
    plt.imshow(figura, cmap='gray')
    plt.title(titulo)
    plt.show()
    plt.close()

def imagen (n, datos):
    imagen = fila_df(datos, n)
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
    fila = fila_df(test, n)
    dist_min = np.linalg.norm(fila - promedios[0])
    k = 1
    while k<10:
        distancia = np.linalg.norm(fila - promedios[k])
        if distancia < dist_min:
            dist_min = distancia
            pred = k
        k+=1
    return pred


def lista_predicciones_promedio(promedios, test):
    predicciones = []
    l = test.shape[0]
    for i in range (0, l, 1):
        pred = prediccion(promedios, test, i)
        predicciones.append(pred)
     
    predicciones = np.array(predicciones)
    return predicciones


test_reducido = df_testeo.head(200)
predicciones = lista_predicciones_promedio(matriz_promedios, test_reducido)


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

"""
EJERCICIO 3
"""

#Dada una matriz A, esta función devuelve el producto entre su transpuesta y ella misma.
def matriz_B(A):
    A_transpuesta = A.T 
    B = A_transpuesta @ A
    
    return B

#Función que, dada una matriz, aplica el método de la potencia para encontrar su primer autovector
def metodo_de_la_potencia(B):
    columnas = B.shape[0]
    x0 = np.random.rand(columnas)
    x0 = x0/np.linalg.norm(x0)
    x1 = B@x0/np.linalg.norm(B@x0, 2)
    
    epsd = np.finfo(np.float64).eps
    while x1@x0 < 1-epsd:
        v = B@x1/np.linalg.norm(B@x1, 2)
        x0 = x1
        x1 = v
    
    return x1

#Funcion que, dados una matriz A y su primer autovector, devuelve el valor de theta correspondiente
def theta(A, v):
    Av = A@v
    sv = np.linalg.norm(Av, 2)
    
    return sv

#Funcion que, dados una matriz A y su primer autovector, devuelve el valor de u correspondiente
def columna_u(A, v):
    Av = A@v
    u = Av/np.linalg.norm(Av, 2)
    
    return u

#Funcion que, dados los valores de A, theta, u y v, devuelve A'
def a_prima(A, sv, u, v):
    
    u = u.reshape(u.shape[0], 1)
    v = v.reshape(1, v.shape[0])
    
    x = (sv*u)@v
    A_prima = A - x
    
    return A_prima
 
#Dados dos vectores u y v en R^n, esta función devuelve la proyección de u sobre v
def proyeccion(u, v):
    v_normalizado = v / np.linalg.norm(v)
    proyec = np.dot(u, v_normalizado) * v_normalizado

    return proyec

#Funcion que, dado un conjunto de vectores de tamaño n, devuelve un nuevo vector de tamaño n
#que será ortogonal a todos ellos y de norma = 1
def vector_ortonormal(vectores):
    #defino un vector cualquiera
    long = len(vectores[0])
    vector_ortogonal = np.random.rand(long)
    
    #busco ortogonalizarlo
    for v in vectores:
        vector_ortogonal = vector_ortogonal - proyeccion(vector_ortogonal, v)
        
    #lo normalizo
    vector_ortonormal = vector_ortogonal/np.linalg.norm(vector_ortogonal, 2)
    
    return vector_ortonormal

#Funcion que, dada una matriz de nxm, con n>m, la extiende a una matriz cuadrada de forma tal
#que sus columnas conformen una base ortonormal de R^n (sea unitaria)
#Se asume que las columnas de la matriz ingresada ya conforman un conjunto ortonormal
def extender_matriz(U):
    filas = U.shape[0]
    columnas = U.shape[1]
    
    #trasponemos la matriz y convertimos el array a una lista de sus vectores columna
    U_transpuesta = U.T
    v_columna = U_transpuesta.tolist()
    
    i = columnas
    while i<filas:
        vector = vector_ortonormal(v_columna)
        U_transpuesta = np.vstack((U_transpuesta, vector))
        v_columna = U_transpuesta.tolist()
        
        i+=1
    
    #Obtenemos la matriz que buscamos, pero transpuesta. Entonces:
    
    matriz_U = U_transpuesta.T
    
    return matriz_U

def descomposicion_SVD(A):
    filas = A.shape[0]
    columnas = A.shape[1]
    
    U_lista = []
    E = np.zeros((filas, columnas))
    V_lista = []
    
    menor = min(filas, columnas)
    for i in range(0, menor):
        B = matriz_B(A)
        v = metodo_de_la_potencia(B)
        u = columna_u(A, v)
        sv = theta(A, v)
        
        U_lista.append(u)
        E[i][i] = sv
        V_lista.append(v)
        
        A = a_prima(A, sv, u, v)
        
    U = np.array(U_lista).T
    V = np.array(V_lista).T
    
    if filas>columnas:
        U = extender_matriz(U)
        
    elif columnas>filas:
        V = extender_matriz(V)
        
    return U, E, V


"""
EJERICIO 4
"""

#ITEM A

#Defino una función que, dado un DataFrame, remueve la primera columna
def sacar_primera_columna(df):
    res = df.iloc[:, 1:]
    return res

#Funcion que dado un dataset, genera una lista conteniendo las 10 matrices con la información de las imágenes
#correspondiente a cada uno de los dígitos
#Las matrices contienen únicamente la información de los pixeles, sin la designación del dígito
#y ahora transpuestas, de manera tal que cada columna representa una imágen. 

def matrices_digitos(datos):
    matrices_digitos = []
    for i in range(0, 10, 1):
        df = df_digito(i, datos)
        df = sacar_primera_columna(df)
        matriz = np.array(df).T
        matrices_digitos.append(matriz)
        
    return matrices_digitos


matrices_digitos = matrices_digitos(datos_reducidos)

#ITEM B

#Dada una lista de matrices, devuelve 3 listas con las matrices correspondientes
#a la descomposición de cada una de ellas
def svd_lista(lista):
    lista_U = []
    lista_E = []
    lista_V = []
    for matriz in lista:
        descomposicion = descomposicion_SVD(matriz)
        U = descomposicion[0]
        lista_U.append(U)
        
        E = descomposicion[1]
        lista_E.append(E)
        
        V = descomposicion[2]
        lista_V.append(V)
        
    return lista_U, lista_E, lista_V

#ITEM C

#Dada una matriz de 784 filas y un n, la función grafica la imagen generada por la columna n de la matriz
def graficar_columna(V, n):
    columna = V[:, n].reshape(28, 28)
    
    plt.imshow(columna, cmap='gray')
    plt.title('Gráfico de columna ')
    plt.show()
    plt.close()
    

matriz_0 = matrices_digitos[0]
descomposicion = descomposicion_SVD(matriz_0)
U_0 = descomposicion[0]
E_0 = descomposicion[1]
graficar_columna(U_0, 2)
m = np.min(U_0)
print(m)

#ITEM E

#Defino una función que, dada una matriz U de mxn y un k entero, devuelve una matriz de mxk 
#de las primeras k columnas de U
def tomar_columnas(U, k):
    matriz = U[:, :k]
    
    return matriz

#Dada una matriz U, la función devuelve la matriz que proyecta ortogonalmente
#sobre la imagen de U
def matriz_de_proyeccion(U):
    U_transpuesta = U.T
    matriz = U @ U_transpuesta
    
    return matriz

#Funcion que, dado un vector v y una matriz U, devuelve la proyección ortogonal
# de v sobre la imagen g
def proyeccion_sobre_imagen(v, U):
    matriz = matriz_de_proyeccion(U)
    proy = matriz @ v
    
    return proy

#Dada una imagen vectorizada x y una matriz U, la función devuelve el residuo, definido como la 
#diferencia entre el vector x y la proyección ortogonal de x sobre la imagen de U
def residuo(x, U):
    residuo = x - proyeccion_sobre_imagen(x, U)
    
    return residuo

#Función que, dada la lista de matrices U correspondientes a todos los dìgitos, una
#imagen vectorizada x y un valor de rango k, devuelve la predicción del dìgito
#para esa imagen en la aproximación del rango k

def prediccion_U(lista_U, x, k):
    #generamos una lista con la distancia para cada uno de los dígitos
    distancias = []
    for i in range(0, 10):
        U = lista_U[i]
        Uk = tomar_columnas(U, k)
        res = residuo(x, Uk)
        distancia = np.linalg.norm(res)
        distancias.append(distancia)
    
    #Una vez creada la lista, buscamos el ìndice de su menor valor
    menor_distancia = min(distancias)
    prediccion = distancias.index(menor_distancia)
    
    return prediccion

matrices_U = svd_lista(matrices_digitos)[0]

#Dada la lista de matrices U, una imagen de testeo y un k entero, la función
#devuelve una lista con las predicciones para la imagen en rango k
def lista_predicciones_U(lista_U, test, k):
    predicciones = []
    l = test.shape[0]
    for i in range(0, l):
        imagen = fila_df(test, i)[1:] #defino el vector con la imagen del digito
        pred = prediccion_U(lista_U, imagen, k)
        predicciones.append(pred)
        
    return predicciones

def lista_precisiones(lista_U, test):
    precisiones = []
    for k in range(1, 6):
        predicciones = lista_predicciones_U(lista_U, test, k)
        prec = precision(test, predicciones)
        precisiones.append(prec)
    
    return precisiones
    
prueba = lista_precisiones(matrices_U, test_reducido)
print(prueba)
