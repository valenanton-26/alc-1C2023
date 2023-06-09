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
df_testeo = datos = pd.read_csv("~/Descargas/mnist_test.csv", names=encabezado)

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

"""
EJERCICIO 3
"""

#Implemetar una funcion en Python que dada una matriz A halle la descomposicion SVD de A,
# por el metodo de la potencia.

# funcion que calcula la matriz B para poder hallar la SVD de la pasada por parametro
def matriz_B(A):
    A_transpuesta = A.T  #transpongo la matriz inicial
    B = A_transpuesta @ A #producto matricial entre la transpuesta y la original
    return B


# funcion que devuelve un vector normalizado
def normalizar_vector(x):
    norma = np.linalg.norm(x)
    return x/norma


# funcion para crear un vector random de norma 1 y de tamaño n
# -> n : cantidad de columnas que tiene la matriz pasada por parametro
def vector_aleatorio_unitario(B):
    x = np.random.rand(B[0].shape[0])
    
    return normalizar_vector(x)
    

# calculo el primer autovector de la matriz A, usando metodo de la potencia
def primer_autovector(A):
    B = matriz_B(A)
    x0 = vector_aleatorio_unitario(B) #nuestro x_(k)
    
    x1 = B @ x0
    x1 = normalizar_vector(x1) # nuestro x_(k+1)

    k = 0
    # x1@x0 < 1 => esta condicion de corte esta dada en el apunte
    # se agrega el k<50 por si la primera no se cumple, y se elije un numero grande para que la aproximacion sea buena
    while(x1@x0 < 1 and k<50): #(x^t_(k+1) * x_(k) < (1 - e))
        #calculo el x que sigue
        x2 = B @ x1
        x2 = normalizar_vector(x2)
        
        # reasigno las variables que conozco con la info anterior
        x0 = x1
        x1 = x2
        
        k += 1
    
    return normalizar_vector(x1)
         
#calculo el autovalor mas grande de la matriz A y el u1, 
# usando el primer autovector de A, calculado con el metodo de la potencia
def primer_theta_u(A , v):
    
    o1 = np.linalg.norm(A@v) #el autovalor
    u1 = normalizar_vector(A@v)
    return o1, u1


# funcion para calcular A' y con esta encontrar la informacion restante para la descomposicion
def a_prima(A, v1, o1, u1):
        
    u11 = np.array(u1).reshape(u1.shape[0], 1)
    v11 = np.array(v1).reshape(1,v1.shape[0])
    
    A_prima = A - o1*(u11@v11)
    return A_prima

#Funcion que, dada una matriz M y un valor n, extiende M a una matriz cuadrada de tamaño n
#con sus vectores siendo una b.o.n del espacio que genera
def extender_matriz(M, n):
    fil = M.shape[0]
    col = M.shape[1]
    
    # si necesito agregarle filas
    while(fil<n):
        b1 = vector_ortonormal(M)
        M = np.vstack((M, b1))
        fil += 1
    
    # si necesito agregarle columnas
    if(col<n):
        M = M.T  #primero traspongo para trabajar mejor
        while(col<n):
            b1 = vector_ortonormal(M)
            M = np.vstack((M, b1))
            col += 1
        M = M.T #devuelvo a la orientacion original
    return M
    
# funcion para calcular un vector ortogonal a los existentes (metodo gram-shmidt)
def vector_ortonormal(M):
    
    # creo un vector random del tamaño necesario
    z = vector_aleatorio_unitario(M)
    # un vector de ceros para ir almacenando mis proyecciones
    v = np.zeros(M.shape[1])
    
    for i in range(0,M.shape[0],1):
        # voy calculando las proyecciones con todos los vectores en M
        v = v + ((M[i] @ z)/(M[i]@M[i]))*M[i]
    
    # el vector ortogonal sera la resta entre el original y sus proyecciones en los vectores de M
    v_0 = z - v
    return v_0

# funcion para extender mi matriz M, parprimer_theta_ua que sea una matriz cuadrada y que sus vectores
# sean una b.o.n. del espacio que genera
# funcion que calcula las matrices U, E y V, necesarias para obtener la descomposicion
# SVD de la matriz pasada por parametro
def descomposicion_SVD(A):
       
    fil = A.shape[0] 
    col = A.shape[1]
    
    menor = 0
    
    if(fil >= col):
        menor = col
    else:
        menor = fil
    
    # calculamos el primer elemento de cada matriz para definir su tamaño
    v1 = primer_autovector(A)
    u1 = primer_theta_u(A, v1)[1]
    o1 = primer_theta_u(A, v1)[0]
    
    U = np.array([u1])
    V = np.array([v1])
    E = np.zeros(A.shape) #matriz de ceros del mismo tamaño que A
    E[0][0] = o1
    
    A = a_prima(A, v1, o1, u1)
    
    # realiza el metodo conocido para el valor menor de filas o columnas
    for i in range(1, menor, 1):
        v = primer_autovector(A)
        u = primer_theta_u(A, v1)[1]
        o = primer_theta_u(A, v1)[0]
        
        
        U = np.vstack((U, u)) #agrego el vector calculado a U
        E[i][i] = o    # modifico la diagonal ii con el avalor hallado
        V = np.vstack((V, v)) #agrego el autovector calculado a V
        
        A = a_prima(A, v, o, u)   # para aplicar el metodo de nuevo debo renombrar A con A'
    
    
    # hay que ver cual de las dos matrices (U o V) hay que rellenar con vectores
    if(fil > col):
        # a U le faltan vectores
        U = extender_matriz(U, fil)
        
    elif (fil < col):
        # a V le faltan vectores
        V = extender_matriz(V, col)

    U = U.T    
    V = V.T
    # devolvemos las metrices calculadas
    return U, E, V

   
"""
#UN EJEMPLO 

# m < n
A = np.array([[3,2,2],[2,3,-2]])

U = descomposicion_SVD(A)[0]
E = descomposicion_SVD(A)[1]
V = descomposicion_SVD(A)[2]
V_tr = V.T

# para verificar que el producto entre las matrices de la descomposicion forma A
A_resul = U @ E @ V_tr

# m > n
A = np.array([[1,1],[1,0],[0,1]])

U = descomposicion_SVD(A)[0]
E = descomposicion_SVD(A)[1]
V = descomposicion_SVD(A)[2]
V_tr = V.T

# para verificar que el producto entre las matrices de la descomposicion forma A
A_resul = U @ E @ V_tr

"""
"""
EJERICIO 4
"""

#ITEM A

#Defino una función que, dado un DataFrame, remueve la primera columna
def sacar_primera_columna(df):
    res = df.iloc[:, 1:]
    return res

#Funcion que dado un dataset, genera un array conteniendo las 10 matrices con la información de las imágenes
#correspondiente a cada uno de los dígitos
def matrices_digitos(datos):
    matrices_digitos = []
    for i in range(0, 10, 1):
        df = sacar_primera_columna(df_digito(i, datos_reducidos))
        matriz = np.array(df)
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
    
    
matriz_0 = matrices_digitos[2].T
desc = descomposicion_SVD(matriz_0)
U_0 = desc[0]
V_0 = desc[2]
E = desc[1]


graficar_columna(U_0, 0)
graficar_columna(U_0, 1)
graficar_columna(U_0, 2)
