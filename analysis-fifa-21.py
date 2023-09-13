# region Configuração inicial

# Importação de todas as bibliotecas necessárias para a análise

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kaggle

# Desabilitar avisos específicos
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Definindo o nome do dataset do Kaggle
dataset_name = 'yagunnersya/fifa-21-messy-raw-dataset-for-cleaning-exploring'

# Definindo o caminho onde salvar o conjunto de dados baixado
download_path = r'C:\Users\linco\Desktop\Outros\Projetos\PowerBI\Portifólio\Datasets'

# Usando a função Kaggle API para baixar e ler o dataset diretamente em um DataFrame do Pandas
kaggle.api.dataset_download_files(dataset_name, unzip=True, path=download_path)
df = pd.read_csv(f'{download_path}/fifa21_raw_data.csv')

# Configurando o tratamento de valores infinitos
pd.set_option('mode.use_inf_as_na', True)  # Para tratar infinitos como NaN

# Configurando o Pandas para exibir todas as colunas
pd.set_option('display.max_columns', None)

# Teste para exibir as primeiras linhas do conjunto de dados
print(df.head())

# Teste de uso de pandas e numpy
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
mean_value = np.mean(data['A'])

# Teste de uso de seaborn e matplotlib
sns.histplot(data['B'])
plt.xlabel('Valores de B')
plt.ylabel('Contagem')
plt.close()

# endregion

# region Cleaning

# Avaliação inicial
df.head()

# Tratamento de dados ausentes e duplicados
df.isnull().sum()
df.duplicated().sum()

# Manipulação de tipo de dados
print(df.dtypes)

# Identificação de outliers

print(df.columns)

# Renomeando a coluna "↓OVA" para "OVA"
df.rename(columns={'↓OVA': 'OVA'}, inplace=True)

# Cálculo por boxplots
plt.figure(figsize=(8, 6))
sns.boxplot(x=df['OVA'])
plt.xlabel('OVA')
plt.title('Boxplot da variável OVA')
plt.close()

# Cálculo por Z-scores
import numpy as np
from scipy import stats

import numpy as np
from scipy import stats

# Calculando z-scores para a coluna 'Age'
z_scores = np.abs(stats.zscore(df['Age']))
limite_z_score = 3
outliers = df[df['Age'] > limite_z_score]

# Visualização de Dispersão
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['OVA'], y=df['POT'], size=outliers['Age'], sizes=(10, 200), hue=outliers['Age'], palette='viridis')

plt.xlabel('OVA')
plt.ylabel('POT')
plt.title('Gráfico de Dispersão: OVA vs POT (Tamanho por Idade)')
plt.legend()
plt.close()


#endregion

# region Exploring

# Resumo Estatístico Inicial
from pydoc import describe
summary = df.describe()
print(summary)

# Histograma de idade
plt.figure(figsize=(8, 6))
sns.histplot(df['Age'], bins=20, kde=True)
plt.xlabel('Idade (Age)')
plt.ylabel('Contagem')
plt.title('Distribuição de Idades')
plt.show()

# Distribuição de overall
plt.figure(figsize=(8, 6))
sns.histplot(df['OVA'], bins=20, kde=True)
plt.xlabel('Overall (OVA)')
plt.ylabel('Contagem')
plt.title('Distribuição de Overall')
plt.show()

# Distribuição de potencial
plt.figure(figsize=(8, 6))
sns.histplot(df['POT'], bins=20, kde=True)
plt.xlabel('Potencial (POT)')
plt.ylabel('Contagem')
plt.title('Distribuição de Potencial')
plt.show()

# Gráfico de Dispersão entre Idade (Age) e Overall (OVA)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Age'], y=df['OVA'])
plt.xlabel('Idade (Age)')
plt.ylabel('Overall (OVA)')
plt.title('Gráfico de Dispersão entre Idade e Overall')
plt.show()


# Gráfico de Dispersão entre Idade (Age) e Potencial (POT)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df['Age'], y=df['POT'])
plt.xlabel('Idade (Age)')
plt.ylabel('Potencial (POT)')
plt.title('Gráfico de Dispersão entre Idade e Potencial')
plt.show()


# Correlação entre as Métricas (Age, OVA, POT)
correlation_matrix = df[['Age', 'OVA', 'POT']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Matriz de Correlação')
plt.show()


# Gráfico de Barras da Nacionalidade (Nationality)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Nationality', order=df['Nationality'].value_counts().index[:10])
plt.xlabel('Nacionalidade (Nationality)')
plt.ylabel('Contagem')
plt.title('Top 10 Nacionalidades dos Jogadores')
plt.xticks(rotation=45)
plt.show()


# Gráfico de Barras das Posições (Positions)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Positions', order=df['Positions'].value_counts().index[:10])
plt.xlabel('Posições (Positions)')
plt.ylabel('Contagem')
plt.title('Top 10 Posições dos Jogadores')
plt.xticks(rotation=45)
plt.show()


# Gráfico de Barras do Clube com mais jogadores em fim de contrato (Team & Contract)
plt.figure(figsize=(12, 6))
sns.countplot(data=df, x='Team & Contract', order=df['Team & Contract'].value_counts().index[:10])
plt.xlabel('Clube (Team & Contract)')
plt.ylabel('Contagem')
plt.title('Top 10 Clubes dos Jogadores')
plt.xticks(rotation=45)
plt.show()


# endregion