import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

#Leitura do arquivo
df = pd.read_csv('europe.csv')
df.head()

Data    = df.iloc[:,1:8]
Country = df.iloc[:,0]
#Estandarizando las variables
X_std = StandardScaler().fit_transform(Data)

def Neurona_Ganadora(Neuronas, Xp):
    
    '''Obtener la neurona ganadadora, o sea, que minimize
    la distancia a la observación Xp'''
    
    k = Neuronas.shape[1]
    Pos_j = 0
    Pos_i = 0
    Dist_Min = +np.inf
    Norma_Minima = 0
    
    '''Slice 3D : [k,i,j]
    k : Capa (de los pesos)
    i : linha
    j : coluna'''
    
    for j in range(k):

        Norma_Linha_Parede = np.linalg.norm(Neuronas[:,:,j].T - Xp , axis=1)
        Norma_Minima       = np.min(Norma_Linha_Parede)

        if Norma_Minima < Dist_Min:
            Dist_Min = Norma_Minima
            Pos_i    = np.argmin(Norma_Linha_Parede)
            Pos_j    = j
            posicion_ganadora = (Pos_i, Pos_j)
            
    return posicion_ganadora, Dist_Min


def Encontrar_Vecinos(Neuronas, Pos_Ganadora, R):
    
    '''Retorna los vecinos (i,j) que están a una distancia <= R de la celda en la posición ganadora (Pos_Ganadora)'''
    
    k = Neuronas.shape[1]
    Lista_Vecinos = list()
    i_g, j_g      = Pos_Ganadora
    Distancia     = 0
    
    for i in range(k):
        for j in range(k):
            if (i == i_g and j == j_g):
                continue
            else:
                Distancia = np.linalg.norm(np.array((i_g, j_g)) - np.array((i, j)))
                if (Distancia <= R):
                    Lista_Vecinos.append((i, j))
    return Lista_Vecinos


def Actualizar_Pesos_Vecinos(Neuronas, Xp, alpha, Pos_Ganadora, Vecinos):
    
    '''Actualiza los pesos de las neuronas vecinas a la celda Pos_Ganadora'''
    
    #Coordenadas de la posicion ganadora 
    i_g, j_g = Pos_Ganadora
    
    DeltaW = 0

    for Punto in Vecinos:
        
        #Coordenadas del vecino 
        i_v, j_v = Punto
        
        DeltaW = np.linalg.norm(np.array((i_g, j_g)) - np.array((i_v, j_v)))
        V = np.exp(-2.0*DeltaW/R)
        Neuronas[:, i_v, j_v] = Neuronas[:, i_v, j_v] + V*alpha*(Xp - Neuronas[:, i_v, j_v])
    
    return Neuronas


#Inicializar la capa de salida (K x K)
K = 10

#Inicializar los pesos de las neuronas con valores aleatorios con distribución uniforme:
Ctd_atributos = X_std.shape[1] #(Area, GDP, Inflation, Life.expect, Military, Pop.growth, Unemployment)
size_matrix = (Ctd_atributos, K, K)
Neuronas = np.zeros(size_matrix)
Neuronas[:,:,:] = np.random.normal(scale=0.25, size=size_matrix)

#Inicializar el radio de vencidad
R = K  #Diap (23/33)
R0 = R #Diap (23/33)

#Factor de aprendizaje
alpha = 0.1

#Inicializar épocas en cero.
max_ctd_epocas = 500 * Ctd_atributos #Diap (23/33)
epoca = 0

while epoca < max_ctd_epocas:
    
    for i in range(0, X_train.shape[0]):
        
        #Seleccionar un registro aleatorio de entrada Xp
        Xp = X_train.sample(1).values
        
        #Encontrar la neurona ganadora k_hat que tenga el vector de pesos W_k_hat más cercano a Xp.
        Pos_Ganadora, Dist_Ganadora = Neurona_Ganadora(Neuronas, Xp)
        i_g, j_g = Pos_Ganadora
        
        #Actualizar los pesos de las conexiones de las neurona ganadora
        Neuronas[:, i_g, j_g] = Neuronas[:, i_g, j_g] + alpha*(Xp - Neuronas[:, i_g, j_g])

        #Actualizar los pesos de las conexiones de las neuronas vecinas
        Vecinos = Encontrar_Vecinos(Neuronas, Pos_Ganadora, R)
        Neuronas = Actualizar_Pesos_Vecinos(Neuronas, Xp, alpha, Pos_Ganadora, Vecinos)
    
    #Incrementar épocas
    epoca += 1
    #Actualizar el radio de vecindad
    R     = (max_ctd_epocas - epoca)*(R0/max_ctd_epocas)
    #Actualizar el factor de aprendizaje
    alpha = 0.1*(1 - epoca/max_ctd_epocas) 
   

# 3. Matriz U 
'''La matriz U tiene, para cada nodo el promedio de la distancia euclidea entre el vector
de pesos del nodo y el vector de pesos de los nodos vecinos'''

Matriz_U = np.zeros(shape=(K,K))

for i in range(K):
    
    for j in range(K):
        
        #w = Neuronas[i,j,:] #Pesos de la celda (i,j)
        w = Neuronas[:,i,j]
        
        suma_distancias = 0
        num_vecinos = 0
        
        #Determinando los vecinos de la celda (i,j)
        
        if i-1 >= 0:  
            #suma_distancias += np.linalg.norm(w - Neuronas[i-1,j,:])
            suma_distancias += np.linalg.norm(w - Neuronas[:,i-1,j])
            num_vecinos += 1
        if i+1 <= K-1: 
            #suma_distancias += np.linalg.norm(w - Neuronas[i+1,j,:])
            suma_distancias += np.linalg.norm(w - Neuronas[:,i+1,j])
            num_vecinos += 1
        if j-1 >= 0:
            #suma_distancias += np.linalg.norm(w - Neuronas[i,j-1,:])
            suma_distancias += np.linalg.norm(w - Neuronas[:,i,j-1])
            num_vecinos += 1
        if j+1 <= K-1:
            #suma_distancias += np.linalg.norm(w - Neuronas[i,j+1,:])
            suma_distancias += np.linalg.norm(w - Neuronas[:,i,j+1])
            num_vecinos += 1
    
        Matriz_U[i][j] = suma_distancias / num_vecinos


#https://en.wikipedia.org/wiki/U-matrix
plt.imshow(Matriz_U, cmap='gray')  
plt.show()