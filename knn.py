# bibliotecas utilizadas
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# sklearn
from sklearn import metrics, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import Bunch
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import ShuffleSplit, StratifiedKFold
from sklearn.neighbors import KNeighborsRegressor


# machine learning algorithm of interest
from sklearn.neighbors import KNeighborsClassifier

def load_data():
    
    # Carregamento do dataset
    data_file = ("./milsa.csv")
    
    # Rótulo dos dados "X"
    xnlabs = ['Funcionario']
    xqlabs = ['EstCivil','Inst','Filhos','Salario', 'Idade','Meses','Regiao']
    xlabs = xnlabs + xqlabs

    # Rótulo dos dados "Y"
    ylabs = ['Salario']

    # Carregar dados para o dataframe do pandas
    df = pd.read_csv(data_file, header=None, names=xlabs)

    
    # Transformar valores
    df['Inst'] = df['Inst'].replace(['1o Grau'],'1')
    df['Inst'] = df['Inst'].replace(['2o Grau'],'2')
    df['Inst'] = df['Inst'].replace(['Superior'],'3')
    df['Inst'] = df['Inst'].astype(int)

    df['EstCivil'] = df['EstCivil'].replace(['casado'],'1')
    df['EstCivil'] = df['EstCivil'].replace(['solteiro'],'0')
    df['EstCivil'] = df['EstCivil'].astype(int)

    df['Filhos'] = df['Filhos'].fillna(0)
    df['Filhos'] = df['Filhos'].astype(int)

    df['Regiao'] = df['Regiao'].replace(['capital'],'0')
    df['Regiao'] = df['Regiao'].replace(['interior'],'1')
    df['Regiao'] = df['Regiao'].replace(['outro'],'2')
    df['Regiao'] = df['Regiao'].astype(int)

    correlation_matrix = df.corr()
    print(correlation_matrix["Salario"])



    df = df.drop("Funcionario", axis=1)
    
    
    
    return Bunch(data   = df,
                 target = df[ylabs],
                 feature_names = xqlabs,
                 target_names  = ylabs)

dataset = load_data()
x = dataset.data.drop("Salario", axis=1)
y = dataset.target

print (x)
print ("-"*20)
print (y.head())

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=12345)


knn_model = KNeighborsRegressor(n_neighbors=3)
knn_model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error
from math import sqrt
train_preds = knn_model.predict(X_train)
mse = mean_squared_error(y_train, train_preds)
print("Raiz quadrada do erro-médio")
rmse = sqrt(mse)
print(rmse)


d = {'EstCivil': [0], 'Inst': [2], 'Filhos': [1], 'Idade': [30], 'Meses': [10], 'Regiao': [0]}
df2 = pd.DataFrame(data=d)

prediction = knn_model.predict(df2)


print("DataFrame de entrada customizada:")
print(df2)
print("O salário previsto para essa entrada é: ")
print(prediction)



test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)

print("RMSE: " + str(rmse))




#Estimativa de salário baseado em meses de empresa e idade
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 4], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)


#Salário real baseado em meses de empresa e idade
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 4], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()

#Estimativa de salário baseado em idade e grau de instrução/escolaridade
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 1], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)


#Salário real baseado em idade e grau de instrução/escolaridade
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 3], X_test.iloc[:, 1], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()
