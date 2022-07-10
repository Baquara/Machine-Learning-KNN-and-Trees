# numbers, stats, plots
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy.stats as stats

# sklearn support
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
    
    # Load the data from this file
    data_file = ("./covid.csv")
    
    # x data labels
    xnlabs = [id]
    xqlabs = ['DataNotificacao','DataCadastro','DataDiagnostico','DataColeta_RT_PCR', 'DataColetaTesteRapido','DataColetaSorologia','DataColetaSorologiaIGG','DataEncerramento','DataObito','Classificacao','Evolucao','CriterioConfirmacao','StatusNotificacao','Municipio','Bairro','FaixaEtaria','IdadeNaDataNotificacao','Sexo','RacaCor','Escolaridade','Gestante','Febre','DificuldadeRespiratoria','Tosse','Coriza','DorGarganta','Diarreia','Cefaleia','ComorbidadePulmao','ComorbidadeCardio','ComorbidadeRenal','ComorbidadeDiabetes','ComorbidadeTabagismo','ComorbidadeObesidade','FicouInternado','ViagemBrasil','ViagemInternacional','ProfissionalSaude','PossuiDeficiencia','MoradorDeRua','ResultadoRT_PCR','ResultadoTesteRapido' ,'ResultadoSorologia','ResultadoSorologia_IGG','TipoTesteRapido']
    																							

    xlabs = xnlabs + xqlabs

    # y data labels
    ylabs = ['ResultadoTesteRapido_Positivo']

    # Load data to dataframe
    df = pd.read_csv(data_file, header=None, skiprows=1, names=xqlabs, sep=';', encoding='latin-1')
    print(df)
    # insert column to start of file
    df.insert(0, "id", range(1, 1 + len(df)))

    df.drop(df.columns[df.columns.str.contains('Data')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('ResultadoSorologia')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('TipoTesteRapido')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('ResultadoRT_PCR')], axis=1, inplace=True)
    df = pd.get_dummies(df)
    df.drop(df.columns[df.columns.str.contains('ResultadoTesteRapido_Não Informado')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('ResultadoTesteRapido_Inconclusivo')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('ResultadoTesteRapido_Negativo')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('_Não')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains('_Nï¿½o')], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("_Descartados")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("_Ignorado")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Classificacao_Confirmados")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("StatusNotificacao_")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("RacaCor")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Sexo")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Escolaridade")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Municipio")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Bairro")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("Gestante_")], axis=1, inplace=True)
    df.drop(df.columns[df.columns.str.contains("CriterioConfirmacao")], axis=1, inplace=True)
    
    
    
    
    
    print(df)


    #df = pd.concat([df['id'], pd.get_dummies(df).iloc[:,1:]], axis = 1)
    

    df = df.drop("id", axis=1)
    correlation_matrix = df.corr()

    print(correlation_matrix["ResultadoTesteRapido_Positivo"])

    return Bunch(data   = df,
                 target = df[ylabs],
                 feature_names = xqlabs,
                 target_names  = ylabs)

dataset = load_data()
x = dataset.data.drop("ResultadoTesteRapido_Positivo", axis=1)
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
rmse = sqrt(mse)
print(rmse)


d = {'Classificacao_Suspeito': [1], 'Classificacao_Sï¿½ndrome Gripal nï¿½o Especificada': [0], 'Evolucao_-': [1], 'Evolucao_Cura': [0], 'Evolucao_ï¿½bito pelo COVID-19': [0], 'FaixaEtaria_0 a 4 anos': [0], 'FaixaEtaria_05 a 9 anos': [0], 'FaixaEtaria_10 a 19 anos': [0], 'FaixaEtaria_20 a 29 anos': [0], 'FaixaEtaria_30 a 39 anos': [1], 'FaixaEtaria_40 a 49 anos': [0], 'FaixaEtaria_50 a 59 anos': [0], 'FaixaEtaria_60 a 69 anos': [0], 'FaixaEtaria_70 a 79 anos': [0], 'FaixaEtaria_80 a 89 anos': [0], 'FaixaEtaria_90 anos ou mais': [0], 'Febre_Sim': [1], 'DificuldadeRespiratoria_Sim': [1], 'Tosse_Sim': [1], 'Coriza_Sim': [0], 'DorGarganta_Sim': [1], 'Diarreia_Sim': [1], 'Cefaleia_Sim': [1], 'ComorbidadePulmao_Sim': [0], 'ComorbidadeCardio_Sim': [0], 'ComorbidadeRenal_Sim': [0], 'ComorbidadeDiabetes_Sim': [0], 'ComorbidadeTabagismo_Sim': [0], 'ComorbidadeObesidade_Sim': [0], 'FicouInternado_Sim': [1], 'ViagemBrasil_Sim': [1], 'ProfissionalSaude_Sim': [1], 'PossuiDeficiencia_Sim': [0], 'MoradorDeRua_Sim': [0]}
df2 = pd.DataFrame(data=d)

prediction = knn_model.predict(df2)

print("-"*100)

print("DataFrame de entrada customizada:")
print(df2)
print("O resultado previsto para essa entrada é: ")
print(prediction)



test_preds = knn_model.predict(X_test)
mse = mean_squared_error(y_test, test_preds)
rmse = sqrt(mse)
print(rmse)

print("a"*100)
print(X_test)
print(X_test.iloc[:, 1])
print(type(X_test))

print("-"*100)



#Estimativa de positivo para quem apresentou problemas de respiração e febre
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 17], X_test.iloc[:, 16], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)


#Dados de positivo para quem apresentou problemas de respiração e febre
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 17], X_test.iloc[:, 16], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()
#Estimativa de positivo para quem apresentou problemas de respiração e viajou
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 17], X_test.iloc[:, 30], c=test_preds, s=100, cmap=cmap
)
f.colorbar(points)

#Dados positivo para quem apresentou problemas de respiração e viajou
cmap = sns.cubehelix_palette(as_cmap=True)
f, ax = plt.subplots()
points = ax.scatter(
    X_test.iloc[:, 17], X_test.iloc[:, 30], c=y_test.iloc[:, 0], s=100, cmap=cmap
)
f.colorbar(points)
plt.show()
