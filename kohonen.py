import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import itertools
import seaborn as sns 
import statistics
from statistics import mode
import pprint


#Lectura del archivo
df = pd.read_csv('europe.csv')
df.head()
Data = df

#Normalizando los datos
Atributos = ['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
Data[Atributos] = StandardScaler().fit_transform(Data[Atributos]) 

X_data = Data.iloc[:,1:] #['Area','GDP','Inflation','Life.expect','Military','Pop.growth','Unemployment']
Y_label = Data.iloc[:,0] #[Country]
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_label, test_size=0.0) #Train_size -> 100%


def most_common(List):
    return(mode(List))

def Neurona_Activada_By_Country(Pais_Neuronas):
    Moda_Neurona_Pais = dict()
    '''Retorna la neurona que fue mas activada por un determinado pais'''
    for key, value in Pais_Neuronas.items():
        Moda_Neurona_Pais[key] = most_common(value)
        
    return Moda_Neurona_Pais
    

def Generate_Capa(K):
    '''Genera un grid Kxk de neuronas en la capa de salida'''
    K = list(range(K))
    index_i_j = list(itertools.product(K, K))
    Capa_Salida = dict()

    for elemento in index_i_j:
        Capa_Salida[elemento] = []

    return Capa_Salida

def Initialize_Weights(Capa, Data):
    #Inicializa los pesos de las neuronas con muestras de los datos de entrada.
    for key, value in Capa.items():
        Capa[key] = Data.sample(1).values
    return Capa

def Neurona_Ganadora(Neuronas, Xp):
    
    '''Obtener la neurona ganadadora, o sea, que minimize
    la distancia a la observación Xp'''
    
    Dist_Min     = +np.inf
    Norma_Minima = 0
    
    #Percorro todos os indices em busca da menor norma.
    for key, value in Neuronas.items():
        
        Norma_Euclidea = np.linalg.norm(Neuronas[key] - Xp)
        #print('key : {0} - Norma : {1}'.format(key, Norma_Euclidea))
        
        if Norma_Euclidea < Dist_Min:
            Dist_Min = Norma_Euclidea
            posicion_ganadora_i_j = key
            
    return posicion_ganadora_i_j, Dist_Min


def Encontrar_Vecinos(Neuronas, Pos_Ganadora, R, K):
    
    '''Retorna los vecinos (i,j) que están a una distancia <= R de la celda en la posición ganadora (Pos_Ganadora)'''
    
    Lista_Vecinos = list()
    i_g, j_g      = Pos_Ganadora
    Distancia     = 0

    for i in range(K):
        for j in range(K):
            if (i == i_g and j == j_g):
                continue
            else:
                #Calculo a distancia da célula ganhadora a todas as células da capa de salida
                #Se a distancia entre elas for <=R então de fato é uma vizinha.
                Distancia = np.linalg.norm(np.array((i_g, j_g)) - np.array((i, j)))
                if (Distancia <= R):
                    Lista_Vecinos.append((i, j))
    return Lista_Vecinos


def Actualizar_Pesos_Vecinos(Neuronas, Xp, alpha, Pos_Ganadora, Vecinos, R):
    
    '''Actualiza los pesos de las neuronas vecinas a la celda Pos_Ganadora'''
    
    #Coordenadas de la posicion ganadora 
    i_g, j_g = Pos_Ganadora
    
    DeltaW = 0

    for Coord_Punto in Vecinos:
        
        #Coordenadas del vecino 
        i_v, j_v = Coord_Punto
        
        DeltaW = np.sqrt((i_g - i_v)**2 + (j_g - j_v)**2)
        V = np.exp(-2.0*DeltaW/R) # o V = 1
        Neuronas[Coord_Punto] = Neuronas[Coord_Punto] + V*alpha*(Xp - Neuronas[Coord_Punto])
    
    return Neuronas


#Inicializar la capa de salida (K x K)
K = 10
Capa_Salida = Generate_Capa(K)
Neuronas    = Initialize_Weights(Capa_Salida, X_train)

#Inicializar el radio de vencidad R(0)
R = K
R0 = R

#Tasa de aprendizaje
alpha0 = 0.2
alpha = alpha0

#Definir el maximo de epocas
MAX_EPOCAS = 700
epoca = 0

#Almacena la celda activada dado un valor del conjunto de entrenamiento.
Lista_Paises = list(Data['Country'])
Pais_Activacion_Neurona = dict()
for pais in Lista_Paises:
    Pais_Activacion_Neurona[pais] = list()
    
#Almacena la cantidad de veces que cada celda de la capa de salida es activada
index_i_j = list(itertools.product(list(range(K)), list(range(K))))
Grid = dict()
for elemento in index_i_j:
    Grid[elemento] = 0

while epoca < MAX_EPOCAS:
    
    for i in range(0, Data.shape[0]):
        
        #Seleccionar un registro aleatorio de entrada Xp
        Entrada = X_train.sample(1)
        Xp      = Entrada.values
        Pais    = y_train[Entrada.index[0]]
        
        #Encontrar la neurona ganadora k_hat que tenga el vector de pesos W_k_hat más cercano a Xp.
        Pos_Ganadora, Dist_Ganadora = Neurona_Ganadora(Neuronas, Xp)
        Grid[Pos_Ganadora] += 1
        
        #Almacena la celda activada por un determinado país.
        Pais_Activacion_Neurona[Pais].append(Pos_Ganadora)
        
        #Actualizar los pesos de la neurona ganadora
        Neuronas[Pos_Ganadora] = Neuronas[Pos_Ganadora] + alpha*(Xp - Neuronas[Pos_Ganadora])

        #Actualizar los pesos de las conexiones de las neuronas vecinas
        Vecinos  = Encontrar_Vecinos(Neuronas, Pos_Ganadora, R, K)
        Neuronas = Actualizar_Pesos_Vecinos(Neuronas, Xp, alpha, Pos_Ganadora, Vecinos, R)
    
    #Incrementar épocas
    epoca += 1
    #Actualizar el radio de vecindad
    R     = (MAX_EPOCAS - epoca)*(R0/MAX_EPOCAS)
    #Actualizar la tasa de aprendizaje
    alpha = alpha0*(1 - epoca/MAX_EPOCAS) 
    
print("C'est fini.")



#Plot heatmap 
coordX = [coord[0] for coord in Grid.keys()]
coordY = [coord[1] for coord in Grid.keys()]
values = [val for val in Grid.values()]
Grid_Dataframe = pd.DataFrame(list(zip(coordX, coordY,values)), columns = ['X', 'Y', 'Value'])
table = Grid_Dataframe.pivot('Y', 'X', 'Value')
fig, ax = plt.subplots(figsize=(6,6))  
sns.heatmap(table,annot=True,fmt='d', linewidths=.5, ax=ax)
ax.invert_yaxis()
plt.show()


Resultado = Neurona_Activada_By_Country(Pais_Activacion_Neurona)
pprint.pprint(Resultado)



# Plot Matriz U 
'''La matriz U tiene, para cada nodo el promedio de la distancia euclidea entre el vector
de pesos del nodo y el vector de pesos de los nodos vecinos'''

Matriz_U = np.zeros(shape=(K,K))

for i in range(K):
    
    for j in range(K):
        
        w = Neuronas[(i,j)]
        suma_distancias = 0
        num_vecinos = 0
        
        #Determinando los vecinos de la celda (i,j)
        "|  x   |(i,j+1)|   x   |"
        "|(i-,j)| (i,j) |(i+1,j)|"
        "|  x   |(i,j+1)|    x  |"
        
        
        #Verificando los limites de la fila 
        if i-1 >= 0:  
            suma_distancias += np.linalg.norm(w - Neuronas[(i-1,j)])
            num_vecinos += 1
        if i+1 <= K-1: 
            suma_distancias += np.linalg.norm(w - Neuronas[(i+1,j)])
            num_vecinos += 1
        
        #Verificando los limites de la columna    
        if j-1 >= 0:
            suma_distancias += np.linalg.norm(w - Neuronas[(i,j-1)])
            num_vecinos += 1
        if j+1 <= K-1:
            suma_distancias += np.linalg.norm(w - Neuronas[(i,j+1)])
            num_vecinos += 1
    
        Matriz_U[i][j] = suma_distancias / num_vecinos


#https://en.wikipedia.org/wiki/U-matrix
c = plt.imshow(Matriz_U, cmap='gray') 
bar = plt.colorbar(c)
plt.show()
