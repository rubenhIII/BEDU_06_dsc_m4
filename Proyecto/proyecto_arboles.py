import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

df = pd.read_csv("./creditcard.csv")

classes = df['Class'].unique()
counts_one = len(df[df['Class']==1])
counts_zero = len(df[df['Class']==0])
counts = [counts_zero, counts_one]
print(counts)

zeros_sample = df[df['Class']==0].sample(492)
ones_sample = df[df['Class']==1]
df2 = zeros_sample._append(ones_sample)

"""
plt.figure(figsize=(12, 8))
plt.bar(df['Class'].unique(), counts)
plt.show()
"""
# Árbol de decisión
# Separa en variable independiente y dependiente
X = df2.drop(['Class'], axis=1)  # Características
y = df2['Class']  # Etiquetas

# Dividimos los datos de prueba (70/30)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=360)

# Creamos un modelo de árbol
Arbol_CA = DecisionTreeClassifier(random_state=360, max_depth = 5)

# Entrenamos el modelo en el conjunto de entrenamiento
Arbol_CA.fit(X_train, y_train)

# predicciones
predicciones = Arbol_CA.predict(X_test)

#Calcula la precisión del modelo en el conjunto de prueba

precision = np.mean(predicciones == y_test)
print(f'Precisión del modelo en el conjunto de prueba: {precision:.2f}')

#Realiza validación cruzada para obtener una estimación más robusta de la precisión
scores = cross_val_score(Arbol_CA, X, y, cv=5)  # 5-fold cross-validation

print(f'Precisión de validación cruzada: {scores.mean():.2f} (+/- {scores.std() * 2:.2f})')

print(classification_report(y_test, predicciones, labels=[0, 1]))
#Recall Positivo -> Sensitivity
#Recall Negativo -> Specificity

plt.figure(figsize=(12, 8))
plot_tree(Arbol_CA, filled=True)
plt.show()
