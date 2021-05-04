import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('europe.csv')
df.head()

#Paso 1 : Tomar un conjunto de datos. Las variables deben estar en las columnas.
Data = df.iloc[:,1:8]
Country = df.iloc[:,0]

#Estandarizando las variables
X_std = StandardScaler().fit_transform(Data)

#Apply PCA method.
pca = PCA(n_components=2)   #Reducir el modelo a 2 componentes principales
X_new = pca.fit_transform(X_std)

#Componentes calculadas
Categorias = ['Area','GDP','Inflation','Life.Expect','Military','Pop.Growth','Unemployment']
PCI_variables_df = pd.DataFrame({'Categorias': Categorias, 'PC1':pca.components_[0,:] , 'PC2':pca.components_[1,:]}  )  
PCI_variables_df

#--------------------------El algoritmo de Oja (Diap 17/19)-------------------------------------

num_inputs = X_std.shape[0] #numero de paises
dimension  = len(X_std[0]) #numero de caracteristicas para cada pais
W          = np.random.normal(scale=0.25, size=(dimension, 1)) #inicializando los pesos
eta        = 0.0001 #factor de aprendizaje
epocas     = 800

while epocas > 0:
    for i in range(num_inputs):
        Ys = np.dot(X_std[i], W_oja) #Linha i vezes coluna de W_oja
        T2 = (X_std[i].reshape(dimension, 1) - Ys*W_oja)
        W_oja = W_oja + eta*Ys*T2
    epocas -= 1

#--------------------------------------------------------------
#Comparación entre resultados via regla de Oja y el metodo PCA
norma = np.linalg.norm(W_oja)
print('Componente Principal (Oja) : ')
print((W_oja/norma).T)

print('Componente Principal (PCA) : ')
print(pca.components_[0,:])
