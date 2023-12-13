import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score

from lazypredict.Supervised import LazyClassifier

df = pd.read_csv("./creditcard_2023.csv")

df.shape
df.head

df.duplicated().sum()

df.columns

df.dtypes

df.isna().sum()

"""# Árbol de decisión"""

# Separa en variable independiente y dependiente
X = df.drop(['Class','id'], axis=1)  # Características
y = df['Class']  # Etiquetas

# Dividimos los datos de prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=360)

# Creamos un modelo de árbol
#Arbol_CA = DecisionTreeClassifier(random_state=360, max_depth = 2)
#Arbol_CA

# Entrenamos el modelo en el conjunto de entrenamiento
#Arbol_CA.fit(X_train, y_train)

# predicciones
#predicciones = Arbol_CA.predict(X_test)
#predicciones

# Calcula la precisión del modelo en el conjunto de prueba

#precision = np.mean(predicciones == y_test)
#print(f'Precisión del modelo en el conjunto de prueba: {precision:.2f}')

# Realiza validación cruzada para obtener una estimación más robusta de la precisión
#scores = cross_val_score(Arbol_CA, X, y, cv=5)  # 5-fold cross-validation

#print(f'Precisión de validación cruzada: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

#plt.figure(figsize=(12, 8))
#plot_tree(Arbol_CA, filled=True)
#plt.show()

"""# Lazypredict

"""

# Inicializa LazyClassifier
clf = LazyClassifier(predictions=True)

# Ajusta el modelo y obtén los resultados
models = clf.fit(X_train, X_test, y_train, y_test)

# Imprime los resultados
print(models)