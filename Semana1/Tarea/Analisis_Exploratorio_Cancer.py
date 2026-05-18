
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuración visual
plt.rcParams['figure.figsize'] = (10,6)

# Cargar dataset de entrenamiento
df = pd.read_csv('wdbc_train.csv')

# Conversión de diagnóstico
df['diagnosis'] = df['diagnosis'].replace({'B':0, 'M':1})

# Eliminar columna ID
df = df.drop(['id'], axis=1)

# Visualizar primeras filas
print(df.head(10))



# Dimensiones
print("Dimensiones del dataset:")
print(df.shape)

# Información general
print("\nInformación general:")
print(df.info())

# Valores nulos
print("\nValores nulos:")
print(df.isnull().sum())



print(df.describe())



plt.figure(figsize=(7,5))

sns.countplot(data=df, x='diagnosis')

plt.xticks([0,1], ['Benigno (0)', 'Maligno (1)'])

plt.title('Distribución de Diagnósticos')
plt.xlabel('Diagnóstico')
plt.ylabel('Cantidad')

plt.show()

print(df['diagnosis'].value_counts())



variables = ['radius1', 'texture1', 'perimeter1', 'area1']

for variable in variables:
    
    plt.figure(figsize=(8,5))
    
    sns.histplot(data=df, x=variable, kde=True)
    
    plt.title(f'Distribución de {variable}')
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    
    plt.show()



variables = ['radius1', 'texture1', 'perimeter1', 'area1']

for variable in variables:
    
    plt.figure(figsize=(8,5))
    
    sns.boxplot(data=df, x='diagnosis', y=variable)
    
    plt.xticks([0,1], ['Benigno', 'Maligno'])
    
    plt.title(f'Boxplot de {variable} por Diagnóstico')
    
    plt.show()



# Matriz de correlación
correlation = df.corr()

plt.figure(figsize=(18,14))

sns.heatmap(correlation, cmap='coolwarm')

plt.title('Mapa de Correlación')

plt.show()



corr_diagnosis = correlation['diagnosis'].sort_values(ascending=False)

print(corr_diagnosis.head(10))

plt.figure(figsize=(10,6))

corr_diagnosis[1:11].plot(kind='bar')

plt.title('Variables más correlacionadas con diagnóstico')
plt.ylabel('Correlación')

plt.show()



variables_pairplot = ['diagnosis', 'radius1', 'texture1', 'perimeter1', 'area1']

sns.pairplot(df[variables_pairplot], hue='diagnosis')

plt.show()
