# -*- coding: utf-8 -*-

# Autor: Denise Mayumi Maeda
# Objetivo: TCC - MBA em Data Science e Analytics USP/ESALQ
# Tema: Aplicação de Modelo de Machine Learning para investigar a saúde mental de profissionais de TI

#%% Informações dos dados

# Informações sobre a saúd mental de profissionais da área de Tecnologia
# Fonte:https://www.kaggle.com/datasets/osmihelp/osmi-mental-health-in-tech-survey-2019
#https://www.kaggle.com/datasets/osmihelp/osmi-mental-health-in-tech-survey-2017
#https://www.kaggle.com/datasets/osmihelp/osmh-2021-mental-health-in-tech-survey-results
#https://www.kaggle.com/datasets/osmihelp/osmi-mental-health-in-tech-survey-2018
#https://www.kaggle.com/datasets/osmihelp/osmi-2020-mental-health-in-tech-survey-results


# In[0.1]: Instalação dos pacotes

!pip install pandas
!pip install numpy
!pip install -U seaborn
!pip install matplotlib
!pip install plotly
!pip install scipy
!pip install statsmodels
!pip install scikit-learn
!pip install playsound
!pip install pingouin
!pip install emojis
!pip install statstests

#%% Carregando os pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
from playsound import playsound # reprodução de sons
import pingouin as pg # outro modo para obtenção de matrizes de correlações
import emojis # inserção de emojis em gráficos
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
import statsmodels.formula.api as smf #estimação do modelo logístico binário
from sklearn.metrics import roc_curve, auc # Função 'roc_curve' do pacote 'metrics' do sklearn
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'


#%% Carregando o dataset

DF_Saude_Mental_Full = pd.read_csv('C:\\Users\\deniz\\OneDrive\\Documentos\\Python Scripts\\TCC\\mental-heath-in-tech-2016_20161114.csv', sep=',')

# Características das variáveis do dataset
DF_Saude_Mental_Full.info()
## Is your primary role within your company related to tech/IT?                      


# Salvando em excel
DF_Saude_Mental_Full.to_excel('C:\\Users\\deniz\\OneDrive\\Documentos\\Python Scripts\\TCC\\test_xx.xlsx', index=False)

#%% verificar nulos
DF_Saude_Mental_Full.isna().sum()

# Remover linhas com qualquer valor NaN
##DF_Saude_Mental_Full = DF_Saude_Mental_Full.dropna()

# Estatísticas univariadas
DF_Saude_Mental_Full.describe()

#%% Alterando os nomes das variáveis (Observar o De Para das Questões)

DF_Saude_Mental_Full.rename(columns={ 'Are you self-employed?' : 'Q1'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'How many employees does your company or organization have?' : 'Q2'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Is your employer primarily a tech company/organization?' : 'Q3'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Is your primary role within your company related to tech/IT?' : 'Q4'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Does your employer provide mental health benefits as part of healthcare coverage?' : 'Q5'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you know the options for mental health care available under your employer-provided coverage?' : 'Q6'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?' : 'Q7'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Does your employer offer resources to learn more about mental health concerns and options for seeking help?' : 'Q8'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?' : 'Q9'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:' : 'Q10'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you think that discussing a mental health disorder with your employer would have negative consequences?' : 'Q11'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you think that discussing a physical health issue with your employer would have negative consequences?' : 'Q12'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you feel comfortable discussing a mental health disorder with your coworkers?' : 'Q13'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?' : 'Q14'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you feel that your employer takes mental health as seriously as physical health?' : 'Q15'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?' : 'Q16'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you have medical coverage (private insurance or state-provided) which includes treatment of  mental health issues?' : 'Q17'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you know local or online resources to seek help for a mental health disorder?' : 'Q18'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to clients or business contacts?' : 'Q19'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have revealed a mental health issue to a client or business contact, do you believe this has impacted you negatively?' : 'Q20'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have been diagnosed or treated for a mental health disorder, do you ever reveal this to coworkers or employees?' : 'Q21'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have revealed a mental health issue to a coworker or employee, do you believe this has impacted you negatively?' : 'Q22'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you believe your productivity is ever affected by a mental health issue?' : 'Q23'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If yes, what percentage of your work time (time performing primary or secondary job functions) is affected by a mental health issue?' : 'Q24'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you have previous employers?' : 'Q25'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have your previous employers provided mental health benefits?' : 'Q26'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Were you aware of the options for mental health care provided by your previous employers?' : 'Q27'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?' : 'Q28'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Did your previous employers provide resources to learn more about mental health issues and how to seek help?' : 'Q29'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?' : 'Q30'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you think that discussing a mental health disorder with previous employers would have negative consequences?' : 'Q31'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you think that discussing a physical health issue with previous employers would have negative consequences?' : 'Q32'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you have been willing to discuss a mental health issue with your previous co-workers?' : 'Q33'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you have been willing to discuss a mental health issue with your direct supervisor(s)?' : 'Q34'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Did you feel that your previous employers took mental health as seriously as physical health?' : 'Q35'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Did you hear of or observe negative consequences for co-workers with mental health issues in your previous workplaces?' : 'Q36'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you be willing to bring up a physical health issue with a potential employer in an interview?' : 'Q37'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Why or why not?' : 'Q38'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Would you bring up a mental health issue with a potential employer in an interview?' : 'Q39'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Why or why not?.1' : 'Q40'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you feel that being identified as a person with a mental health issue would hurt your career?' : 'Q41'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?' : 'Q42'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'How willing would you be to share with friends and family that you have a mental illness?' : 'Q43'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?' : 'Q44'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have your observations of how another individual who discussed a mental health disorder made you less likely to reveal a mental health issue yourself in your current workplace?' : 'Q45'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you have a family history of mental illness?' : 'Q46'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have you had a mental health disorder in the past?' : 'Q47'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you currently have a mental health disorder?' : 'Q48'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If yes, what condition(s) have you been diagnosed with?' : 'Q49'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If maybe, what condition(s) do you believe you have?' : 'Q50'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have you been diagnosed with a mental health condition by a medical professional?' : 'Q51'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If so, what condition(s) were you diagnosed with?' : 'Q52'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Have you ever sought treatment for a mental health issue from a mental health professional?' : 'Q53'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have a mental health issue, do you feel that it interferes with your work when being treated effectively?' : 'Q54'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'If you have a mental health issue, do you feel that it interferes with your work when NOT being treated effectively?' : 'Q55'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What is your age?' : 'Q56'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What is your gender?' : 'Q57'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What country do you live in?' : 'Q58'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What US state or territory do you live in?' : 'Q59'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What country do you work in?' : 'Q60'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'What US state or territory do you work in?' : 'Q61'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Which of the following best describes your work position?' : 'Q62'}, inplace=True)
DF_Saude_Mental_Full.rename(columns={ 'Do you work remotely?' : 'Q63'}, inplace=True)

#%% --------------------------------------------
# Seleção do público inicial
# ----------------------------------------------


# Analisar as observações distintas das vvariáveis, exceto a idade. No excel

DF_SM_Colunas_ = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q46', 'Q47', 'Q48', 'Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55', 'Q56', 'Q57', 'Q60', 'Q62', 'Q63','Q58', 'Q61', 'Q59']
DF_SM_Colunas = DF_Saude_Mental_Full[DF_SM_Colunas_]

DF_SM_Distinct = DF_SM_Colunas.drop_duplicates(subset=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q9', 'Q10', 'Q11', 'Q12', 'Q13', 'Q14', 'Q15', 'Q16', 'Q17', 'Q18', 'Q19', 'Q20', 'Q21', 'Q22', 'Q23', 'Q24', 'Q25', 'Q26', 'Q27', 'Q28', 'Q29', 'Q30', 'Q31', 'Q32', 'Q33', 'Q34', 'Q35', 'Q36', 'Q37', 'Q38', 'Q39', 'Q40', 'Q41', 'Q42', 'Q43', 'Q44', 'Q45', 'Q46', 'Q47', 'Q48', 'Q49', 'Q50', 'Q51', 'Q52', 'Q53', 'Q54', 'Q55', 'Q57', 'Q58', 'Q59', 'Q60', 'Q61', 'Q62', 'Q63'], keep=False, inplace=False, ignore_index=False)

# Salvando em excel
DF_SM_Distinct.to_excel('C:\\Users\\deniz\\OneDrive\\Documentos\\Python Scripts\\TCC\\DF_SM_Distinct.xlsx', index=False)


#%% -----------------------------------------
# Tratando as variáveis a serem consideradas
# --------------------------------------------

DF_Saude_Mental_Full['Q4'].unique() 

DF_Saude_Mental = DF_SM_Colunas.copy() 



## How many employees does your company or organization have?
DF_Saude_Mental['Q2'].unique() 
'''
'1-5'
'6-25'
'26-100'
'100-500'
'500-1000'
'More than 1000'
nan
'''

# Substituir 
DF_Saude_Mental['Q2']  = DF_Saude_Mental['Q2'].str.replace('-', '_', regex=False)
DF_Saude_Mental['Q2']  = DF_Saude_Mental['Q2'].str.replace(' ', '_', regex=False)

## Does your employer provide mental health benefits as part of healthcare coverage?
DF_Saude_Mental['Q5'].unique() # ['Not eligible for coverage / N/A', 'No', nan, 'Yes', "I don't know"]

# Removendo o apóstrofo
DF_Saude_Mental['Q5']  = DF_Saude_Mental['Q5'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q5']  =   np.where(np.isin(DF_Saude_Mental['Q5'], [ 'Yes'])                             , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q5'], [ 'No'])                              , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q5'], [ 'I dont know'])                     , 'A3',
                           np.where(np.isin(DF_Saude_Mental['Q5'], [ 'Not eligible for coverage / N/A']) , 'A4',
                           np.nan))))


## Do you know the options for mental health care available under your employer-provided coverage?
DF_Saude_Mental['Q6'].unique() #[nan, 'Yes', 'I am not sure', 'No'
# Removendo o apóstrofo
DF_Saude_Mental['Q6']  = DF_Saude_Mental['Q6'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q6']  =   np.where(np.isin(DF_Saude_Mental['Q6'], [ 'Yes'])          , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q6'], [ 'No'])           , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q6'], [ 'I am not sure']), 'A3',
                           np.nan)))

#$ Has your employer ever formally discussed mental health (for example, as part of a wellness campaign or other official communication)?
DF_Saude_Mental['Q7'].unique() # 'No', 'Yes', nan, "I don't know"

# Removendo o apóstrofo
DF_Saude_Mental['Q7']  = DF_Saude_Mental['Q7'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q7']  =  np.where(np.isin(DF_Saude_Mental['Q7'], [ 'Yes'])        , 'A1',
                          np.where(np.isin(DF_Saude_Mental['Q7'], [ 'No'])         , 'A2',
                          np.where(np.isin(DF_Saude_Mental['Q7'], [ 'I dont know']), 'A3',
                          np.nan)))

#$ Does your employer offer resources to learn more about mental health concerns and options for seeking help?
DF_Saude_Mental['Q8'].unique() # 'No', 'Yes', nan, "I don't know"

# Removendo o apóstrofo
DF_Saude_Mental['Q8']  = DF_Saude_Mental['Q8'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q8']  =  np.where(np.isin(DF_Saude_Mental['Q8'], [ 'Yes'])        , 'A1',
                          np.where(np.isin(DF_Saude_Mental['Q8'], [ 'No'])         , 'A2',
                          np.where(np.isin(DF_Saude_Mental['Q8'], [ 'I dont know']), 'A3',
                          np.nan)))

## Is your anonymity protected if you choose to take advantage of mental health or substance abuse treatment resources provided by your employer?
DF_Saude_Mental['Q9'].unique() # "I don't know", 'Yes', nan, 'No'

# Removendo o apóstrofo
DF_Saude_Mental['Q9']  = DF_Saude_Mental['Q9'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q9']  =  np.where(np.isin(DF_Saude_Mental['Q9'], [ 'Yes'])        , 'A1',
                          np.where(np.isin(DF_Saude_Mental['Q9'], [ 'No'])         , 'A2',
                          np.where(np.isin(DF_Saude_Mental['Q9'], [ 'I dont know']), 'A3',
                          np.nan)))

## If a mental health issue prompted you to request a medical leave from work, asking for that leave would be:
DF_Saude_Mental['Q10'].unique()

# Removendo o apóstrofo
DF_Saude_Mental['Q10']  = DF_Saude_Mental['Q10'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q10']  =   np.where(np.isin(DF_Saude_Mental['Q10'], [ 'Very easy'])                  , 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q10'], [ 'Somewhat easy'])              , 'A2',
                            np.where(np.isin(DF_Saude_Mental['Q10'], [ 'Neither easy nor difficult']) , 'A3',
                            np.where(np.isin(DF_Saude_Mental['Q10'], [ 'Somewhat difficult'])         , 'A4',
                            np.where(np.isin(DF_Saude_Mental['Q10'], [ 'Very difficult'])             , 'A5',
                            np.where(np.isin(DF_Saude_Mental['Q10'], [ 'I dont know'])                , 'A10',
                            np.nan))))))

## Do you think that discussing a mental health disorder with your employer would have negative consequences?
DF_Saude_Mental['Q11'].unique() #'Maybe', nan, 'Yes', 'No']

DF_Saude_Mental['Q11']  =  np.where(np.isin(DF_Saude_Mental['Q11'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q11'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q11'], [ 'Maybe']), 'A3',
                           np.nan)))

## Do you think that discussing a physical health issue with your employer would have negative consequences?
DF_Saude_Mental['Q12'].unique() #'Maybe', nan, 'Yes', 'No']

DF_Saude_Mental['Q12']  =  np.where(np.isin(DF_Saude_Mental['Q12'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q12'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q12'], [ 'Maybe']), 'A3',
                           np.nan)))


## Would you feel comfortable discussing a mental health disorder with your coworkers?
DF_Saude_Mental['Q13'].unique() #'Maybe', nan, 'Yes', 'No']

DF_Saude_Mental['Q13']  =  np.where(np.isin(DF_Saude_Mental['Q13'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q13'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q13'], [ 'Maybe']), 'A3',
                           np.nan)))

## Would you feel comfortable discussing a mental health disorder with your direct supervisor(s)?
DF_Saude_Mental['Q14'].unique() #'Yes', 'Maybe', nan, 'No'

DF_Saude_Mental['Q14']  =  np.where(np.isin(DF_Saude_Mental['Q14'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q14'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q14'], [ 'Maybe']), 'A3',
                           np.nan)))

## Do you feel that your employer takes mental health as seriously as physical health?
DF_Saude_Mental['Q15'].unique() #"I don't know", 'Yes', nan, 'No'

# Removendo o apóstrofo
DF_Saude_Mental['Q15']  = DF_Saude_Mental['Q15'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q15']  =  np.where(np.isin(DF_Saude_Mental['Q15'], [ 'Yes'])        , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q15'], [ 'No'])         , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q15'], [ 'I dont know']), 'A3',
                           np.nan)))

## Have you heard of or observed negative consequences for co-workers who have been open about mental health issues in your workplace?
DF_Saude_Mental['Q16'].unique() #'No', nan, 'Yes'

DF_Saude_Mental['Q16']  = np.where(np.isin(DF_Saude_Mental['Q16'], [ 'Yes'])  , 1,
                          np.where(np.isin(DF_Saude_Mental['Q16'], ['No'])    , 0,
                          np.nan))

## Do you have medical coverage (private insurance or state-provided) which includes treatment of Â mental health issues?
DF_Saude_Mental['Q17'].unique() # Q17 até Q24, sem respostas


## Do you have previous employers?
DF_Saude_Mental['Q25'].unique() #[1, 0]

## Have your previous employers provided mental health benefits?
DF_Saude_Mental['Q26'].unique()

DF_Saude_Mental['Q26']  = DF_Saude_Mental['Q26'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q26']  =   np.where(np.isin(DF_Saude_Mental['Q26'], [ 'Yes, they all did']), 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q26'], [ 'Some did'])         , 'A2',
                            np.where(np.isin(DF_Saude_Mental['Q26'], [ 'I dont know'])      , 'A3',
                            np.where(np.isin(DF_Saude_Mental['Q26'], [ 'No, none did'])     , 'A4',
                            np.nan))))

## Were you aware of the options for mental health care provided by your previous employers?
DF_Saude_Mental['Q27'].unique()

DF_Saude_Mental['Q27']  = DF_Saude_Mental['Q27'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q27']  =   np.where(np.isin(DF_Saude_Mental['Q27'], [ 'Yes, I was aware of all of them']), 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q27'], [ 'I was aware of some'])            , 'A2',
                            np.where(np.isin(DF_Saude_Mental['Q27'], [ 'No, I only became aware later'])  , 'A3',
                            np.where(np.isin(DF_Saude_Mental['Q27'], [ 'N/A (not currently aware)'])      , 'A4',
                            np.nan))))

## Did your previous employers ever formally discuss mental health (as part of a wellness campaign or other official communication)?

DF_Saude_Mental['Q28'].unique()

DF_Saude_Mental['Q28']  = DF_Saude_Mental['Q28'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q28']  =   np.where(np.isin(DF_Saude_Mental['Q28'], [ 'Some did'])         , 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q28'], [ 'Yes, they all did']), 'A2',         
                            np.where(np.isin(DF_Saude_Mental['Q28'], [ 'I dont know'])      , 'A3',              
                            np.where(np.isin(DF_Saude_Mental['Q28'], [ 'None did'])         , 'A4',
                            np.nan))))

## Did your previous employers provide resources to learn more about mental health issues and how to seek help?

DF_Saude_Mental['Q29'].unique()

DF_Saude_Mental['Q29']  = DF_Saude_Mental['Q29'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q29']  =   np.where(np.isin(DF_Saude_Mental['Q29'], [ 'Some did'])         , 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q29'], [ 'Yes, they all did']), 'A2',         
                            np.where(np.isin(DF_Saude_Mental['Q29'], [ 'I dont know'])      , 'A3',
                            np.where(np.isin(DF_Saude_Mental['Q29'], [ 'None did'])         , 'A4',
                            np.nan))))

## Was your anonymity protected if you chose to take advantage of mental health or substance abuse treatment resources with previous employers?

DF_Saude_Mental['Q30'].unique()

DF_Saude_Mental['Q30']  = DF_Saude_Mental['Q30'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q30']  =    np.where(np.isin(DF_Saude_Mental['Q30'], [ 'Sometimes']), 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q30'], [ 'Yes, always']), 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q30'], [ 'I dont know']), 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q30'], [ 'No']), 'A4',
                             np.nan))))

## Do you think that discussing a mental health disorder with previous employers would have negative consequences?
DF_Saude_Mental['Q31'].unique()

DF_Saude_Mental['Q31']  = DF_Saude_Mental['Q31'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q31']  =    np.where(np.isin(DF_Saude_Mental['Q31'], [ 'Some of them']), 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q31'], [ 'Yes, all of them']), 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q31'], [ 'I dont know']), 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q31'], [ 'None of them']), 'A4',
                             np.nan))))

## Did you feel that your previous employers took mental health as seriously as physical health?
DF_Saude_Mental['Q35'].unique()

DF_Saude_Mental['Q35']  = DF_Saude_Mental['Q35'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q35']  =    np.where(np.isin(DF_Saude_Mental['Q35'], [ 'Yes, they all did'])   , 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q35'], [ 'Some did'])            , 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q35'], [ 'I dont know'])         , 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q35'], [ 'None did'])            , 'A4',
                             np.nan))))

## Do you feel that being identified as a person with a mental health issue would hurt your career?
DF_Saude_Mental['Q41'].unique()# 'Maybe', "No, I don't think it would", 'Yes, I think it would','No, it has not', 'Yes, it has'

DF_Saude_Mental['Q41']  = DF_Saude_Mental['Q41'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q41']  =    np.where(np.isin(DF_Saude_Mental['Q41'], [ 'Yes, I think it would'])    , 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q41'], [ 'Yes, it has'])              , 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q41'], [ 'Maybe'])                    , 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q41'], [ 'No, I dont think it would']), 'A4',
                             np.nan))))


## Do you think that team members/co-workers would view you more negatively if they knew you suffered from a mental health issue?
DF_Saude_Mental['Q42'].unique()#"No, I don't think they would", 'Maybe', 'Yes, they do', 'Yes, I think they would', 'No, they do not'

DF_Saude_Mental['Q42']  = DF_Saude_Mental['Q42'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q42']  =    np.where(np.isin(DF_Saude_Mental['Q42'], [ 'Yes, I think they would'])     , 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q42'], [ 'Yes, they do'])                , 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q42'], [ 'Maybe'])                       , 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q42'], [ 'No, I dont think they would']) , 'A4',
                             np.where(np.isin(DF_Saude_Mental['Q42'], [ 'No, they do not'])             , 'A5',
                             np.nan)))))

## How willing would you be to share with friends and family that you have a mental illness?
DF_Saude_Mental['Q43'].unique()# 'Somewhat open', 'Neutral','Not applicable to me (I do not have a mental illness)', 'Very open', 'Not open at all', 'Somewhat not open'

DF_Saude_Mental['Q43']  = DF_Saude_Mental['Q43'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q43']  =    np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Somewhat open'])       , 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Very open'])           , 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Neutral'])             , 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Somewhat not open'])   , 'A4',
                             np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Not open at all'])     , 'A5',
                             np.where(np.isin(DF_Saude_Mental['Q43'], [ 'Not applicable to me (I do not have a mental illness)']), 'A6',
                             np.nan))))))

## Have you observed or experienced an unsupportive or badly handled response to a mental health issue in your current or previous workplace?
DF_Saude_Mental['Q44'].unique()#'No', 'Maybe/Not sure', 'Yes, I experienced', 'Yes, I observed', nan

DF_Saude_Mental['Q44']  = DF_Saude_Mental['Q44'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q44']  =    np.where(np.isin(DF_Saude_Mental['Q44'], [ 'Yes, I observed'])   , 'A1',
                             np.where(np.isin(DF_Saude_Mental['Q44'], [ 'Yes, I experienced']), 'A2',
                             np.where(np.isin(DF_Saude_Mental['Q44'], [ 'Maybe/Not sure'])    , 'A3',
                             np.where(np.isin(DF_Saude_Mental['Q44'], [ 'No'])                , 'A4',
                             np.nan))))

## Do you have a family history of mental illness?
DF_Saude_Mental['Q46'].unique()#'No', 'Yes', "I don't know"

# Removendo o apóstrofo
DF_Saude_Mental['Q46']  = DF_Saude_Mental['Q46'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q46']  =  np.where(np.isin(DF_Saude_Mental['Q46'], [ 'Yes'])        , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q46'], [ 'No'])         , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q46'], [ 'I dont know']), 'A3',
                           np.nan)))

## Have you had a mental health disorder in the past?
DF_Saude_Mental['Q47'].unique()#'Yes', 'Maybe', 'No']
DF_Saude_Mental['Q47']  = DF_Saude_Mental['Q47'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q47']  =  np.where(np.isin(DF_Saude_Mental['Q47'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q47'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q47'], [ 'Maybe']), 'A3',
                           np.nan)))

## Do you currently have a mental health disorder?
DF_Saude_Mental['Q48'].unique()#'No', 'Yes', 'Maybe'

DF_Saude_Mental['Q48'] = DF_Saude_Mental['Q48'].str.strip().str.lower()
DF_Saude_Mental['Q48']  =  np.where(np.isin(DF_Saude_Mental['Q48'], [ 'Yes'])  , 'A1',
                           np.where(np.isin(DF_Saude_Mental['Q48'], [ 'No'])   , 'A2',
                           np.where(np.isin(DF_Saude_Mental['Q48'], [ 'Maybe']), 'A3',
                           np.nan)))

## If yes, what condition(s) have you been diagnosed with?
DF_Saude_Mental['Q49'].unique()#

DF_Saude_Mental['Q49'] = DF_Saude_Mental['Q49'].str.strip().str.lower()

DF_Saude_Mental['Q49'] = np.where( DF_Saude_Mental['Q49'].str.contains('anxiety'          , case=False, na=False), 'Transtorno_Ansiedade', 
                         #np.where( DF_Saude_Mental['Q49'].str.contains('personality'      , case=False, na=False), 'Transtorno_Personalidade',
                         np.where( DF_Saude_Mental['Q49'].str.contains('mood disorder'    , case=False, na=False), 'Transtorno_Humor',
                         np.where( DF_Saude_Mental['Q49'].str.contains('attention deficit', case=False, na=False), 'Transtorno_Deficit_Atencoo_Hiperatividade',
                         np.where( DF_Saude_Mental['Q49'].str.contains('peasonal affectiv', case=False, na=False), 'Transtorno_Afetivo_Sazonal',
                         np.where( DF_Saude_Mental['Q49'].str.contains('pdd'              , case=False, na=False), 'PDD',
                         np.where( DF_Saude_Mental['Q49'].str.contains('post-traumatic'   , case=False, na=False), 'Transtorno_Estresse_Pos_Traumatico',
                         np.where( DF_Saude_Mental['Q49'].str.contains('depression'       , case=False, na=False), 'Depressao',
                         np.where( DF_Saude_Mental['Q49'].str.contains('gender identity'  , case=False, na=False), 'Transtorno_Identidade_Genero',
                         np.where( DF_Saude_Mental['Q49'].str.contains('substance use'    , case=False, na=False), 'Transtorno_Uso_Substancias',
                         np.where( DF_Saude_Mental['Q49'].str.contains('addictive'        , case=False, na=False), 'Transtorno_Viciante',
                         np.where( DF_Saude_Mental['Q49'].str.contains('stress response'  , case=False, na=False), 'Sindromes_Resposta_Estresse',
                         np.where( DF_Saude_Mental['Q49'].str.contains('autism '          , case=False, na=False), 'Autismo',
                         np.where( DF_Saude_Mental['Q49'].str.contains('obsessive compuls', case=False, na=False), 'Transtorno_Obsessivo_Compulsivo',
                         np.where( DF_Saude_Mental['Q49'].str.contains('psychotic'        , case=False, na=False), 'Transtorno_Psicotico',
                         np.where( DF_Saude_Mental['Q49'].str.contains('eating'           , case=False, na=False), 'Transtorno_Alimentar',
                         np.where( DF_Saude_Mental['Q49'].str.contains('asperger'         , case=False, na=False), 'Sindrome_Asperger',
                         np.where( DF_Saude_Mental['Q49'].str.contains('add'              , case=False, na=False), 'ADD',
                         np.where( DF_Saude_Mental['Q49'].str.contains('schizotypal'      , case=False, na=False), 'Transtorno_Personalidade_Esquizotipica',
                         np.where( DF_Saude_Mental['Q49'].str.contains('transgender'      , case=False, na=False), 'Transgender',         
                         np.nan)))))))))))))))))))

## If maybe, what condition(s) do you believe you have?
DF_Saude_Mental['Q50'] = DF_Saude_Mental['Q50'].str.strip().str.lower()

DF_Saude_Mental['Q50'] = np.where( DF_Saude_Mental['Q50'].str.contains('anxiety'          , case=False, na=False), 'Transtorno_Ansiedade', 
                         np.where( DF_Saude_Mental['Q50'].str.contains('personality'      , case=False, na=False), 'Transtorno_Personalidade',
                         np.where( DF_Saude_Mental['Q50'].str.contains('mood disorder'    , case=False, na=False), 'Transtorno_Humor',
                         np.where( DF_Saude_Mental['Q50'].str.contains('attention deficit', case=False, na=False), 'Transtorno_Deficit_Atencoo_Hiperatividade',
                         np.where( DF_Saude_Mental['Q50'].str.contains('peasonal affectiv', case=False, na=False), 'Transtorno_Afetivo_Sazonal',
                         np.where( DF_Saude_Mental['Q50'].str.contains('pdd'              , case=False, na=False), 'PDD',
                         np.where( DF_Saude_Mental['Q50'].str.contains('post-traumatic'   , case=False, na=False), 'Transtorno_Estresse_Pos_Traumatico',
                         np.where( DF_Saude_Mental['Q50'].str.contains('depression'       , case=False, na=False), 'Depressao',
                         np.where( DF_Saude_Mental['Q50'].str.contains('gender identity'  , case=False, na=False), 'Transtorno_Identidade_Genero',
                         np.where( DF_Saude_Mental['Q50'].str.contains('substance use'    , case=False, na=False), 'Transtorno_Uso_Substancias',
                         np.where( DF_Saude_Mental['Q50'].str.contains('addictive'        , case=False, na=False), 'Transtorno_Viciante',
                         np.where( DF_Saude_Mental['Q50'].str.contains('stress response'  , case=False, na=False), 'Sindromes_Resposta_Estresse',
                         np.where( DF_Saude_Mental['Q50'].str.contains('autism '          , case=False, na=False), 'Autismo',
                         np.where( DF_Saude_Mental['Q50'].str.contains('obsessive compuls', case=False, na=False), 'Transtorno_Obsessivo_Compulsivo',
                         np.where( DF_Saude_Mental['Q50'].str.contains('psychotic'        , case=False, na=False), 'Transtorno_Psicotico',
                         np.where( DF_Saude_Mental['Q50'].str.contains('eating'           , case=False, na=False), 'Transtorno_Alimentar',
                         np.where( DF_Saude_Mental['Q50'].str.contains('asperger'         , case=False, na=False), 'Sindrome_Asperger',
                         np.where( DF_Saude_Mental['Q50'].str.contains('add'              , case=False, na=False), 'ADD',
                         np.where( DF_Saude_Mental['Q50'].str.contains('schizotypal'      , case=False, na=False), 'Transtorno_Personalidade_Esquizotipica',
                         np.where( DF_Saude_Mental['Q50'].str.contains('transgender'      , case=False, na=False), 'Transgender',          
                         np.nan))))))))))))))))))))

## Variavel em estudo, vaiável resposta: Q51 - Have you been diagnosed with a mental health condition by a medical professional?
DF_Saude_Mental['Q51'].unique() ## Yes, No

Diagnistico_Counts = DF_Saude_Mental['Q51'].value_counts()
print(Diagnistico_Counts)


## Normalizar as entradas (tudo minúsculo e sem espaços extras)
DF_Saude_Mental['Q51'] = DF_Saude_Mental['Q51'].str.strip().str.lower()

#DF_Saude_Mental['Q51']  = {'yes': 1, 'no': 0}

DF_Saude_Mental['Q51']  = np.where(np.isin(DF_Saude_Mental['Q51'], ['yes']), 1,
                          np.where(np.isin(DF_Saude_Mental['Q51'], ['no']) , 0,
                          np.nan))


## Normalizar as entradas (tudo minúsculo e sem espaços extras)
DF_Saude_Mental['Q52'] = DF_Saude_Mental['Q52'].str.strip().str.lower()

DF_Saude_Mental['Q52'] = np.where( DF_Saude_Mental['Q52'].str.contains('anxiety'          , case=False, na=False), 'Transtorno_Ansiedade', 
                         np.where( DF_Saude_Mental['Q52'].str.contains('personality'      , case=False, na=False), 'Transtorno_Personalidade',
                         np.where( DF_Saude_Mental['Q52'].str.contains('mood disorder'    , case=False, na=False), 'Transtorno_Humor',
                         np.where( DF_Saude_Mental['Q52'].str.contains('attention deficit', case=False, na=False), 'Transtorno_Deficit_Atencoo_Hiperatividade',
                         np.where( DF_Saude_Mental['Q52'].str.contains('peasonal affectiv', case=False, na=False), 'Transtorno_Afetivo_Sazonal',
                         np.where( DF_Saude_Mental['Q52'].str.contains('pdd'              , case=False, na=False), 'PDD',
                         np.where( DF_Saude_Mental['Q52'].str.contains('post-traumatic'   , case=False, na=False), 'Transtorno_Estresse_Pos_Traumatico',
                         np.where( DF_Saude_Mental['Q52'].str.contains('depression'       , case=False, na=False), 'Depressao',
                         np.where( DF_Saude_Mental['Q52'].str.contains('gender identity'  , case=False, na=False), 'Transtorno_Identidade_Genero',
                         np.where( DF_Saude_Mental['Q52'].str.contains('substance use'    , case=False, na=False), 'Transtorno_Uso_Substancias',
                         np.where( DF_Saude_Mental['Q52'].str.contains('addictive'        , case=False, na=False), 'Transtorno_Viciante',
                         np.where( DF_Saude_Mental['Q52'].str.contains('stress response'  , case=False, na=False), 'Sindromes_Resposta_Estresse',
                         np.where( DF_Saude_Mental['Q52'].str.contains('autism '          , case=False, na=False), 'Autismo',
                         np.where( DF_Saude_Mental['Q52'].str.contains('obsessive compuls', case=False, na=False), 'Transtorno_Obsessivo_Compulsivo',
                         np.where( DF_Saude_Mental['Q52'].str.contains('psychotic'        , case=False, na=False), 'Transtorno_Psicotico',
                         np.where( DF_Saude_Mental['Q52'].str.contains('eating'           , case=False, na=False), 'Transtorno_Alimentar',
                         np.where( DF_Saude_Mental['Q52'].str.contains('asperger'         , case=False, na=False), 'Sindrome_Asperger',
                         np.where( DF_Saude_Mental['Q52'].str.contains('add'              , case=False, na=False), 'ADD',
                         np.where( DF_Saude_Mental['Q52'].str.contains('schizotypal'      , case=False, na=False), 'Transtorno_Personalidade_Esquizotipica',
                         np.where( DF_Saude_Mental['Q52'].str.contains('transgender'      , case=False, na=False), 'Transgender',          
                         np.nan))))))))))))))))))))


## Have you ever sought treatment for a mental health issue from a mental health professional?
DF_Saude_Mental['Q53'].unique()# 0, 1

# ----------------------------------------------
## What is your gender?
# ----------------------------------------------

DF_Saude_Mental['Q57'].unique() 

# Normalizar as entradas (tudo minúsculo e sem espaços extras)
DF_Saude_Mental['Q57'] = DF_Saude_Mental['Q57'].str.strip().str.lower()

# Substituir o ponto por outro branco
DF_Saude_Mental['Q57']  = DF_Saude_Mental['Q57'] .str.replace('.', '', regex=False)

# Removendo o apóstrofo
DF_Saude_Mental['Q57']  = DF_Saude_Mental['Q57'].str.replace("'", "", regex=False)

# Tratar exceção
DF_Saude_Mental['Q57']  = DF_Saude_Mental['Q57'] .str.replace('im a man why didnt you make this a drop down question you should of asked sex? and i would of answered yes please seriously how much text can this take?', 'm', regex=False)

# Criar Variável Sexo
DF_Saude_Mental['Q57']  = np.where(np.isin(DF_Saude_Mental['Q57'], [ 'female', 'f', 'woman', 'fem', 'female/woman', 'cis-woman', 'cis female', 'cisgender female'])    , 'F',
                          np.where(np.isin(DF_Saude_Mental['Q57'], ['male', 'm', 'man', 'm|', 'sex is male', 'cis male',  'cis man', 'male (cis)', 'cisdude', 'dude']) , 'M',
                          np.nan))


## Which of the following best describes your work position?
DF_Saude_Mental['Q62'].unique()# 

DF_Saude_Mental['Q62']  = DF_Saude_Mental['Q62'].str.replace("'", "", regex=False)

DF_Saude_Mental['Q62'] =   np.where( DF_Saude_Mental['Q62'].str.contains('Bbck-end Developer'       , case=False, na=False), 'Back_end_Developer',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Designer'                 , case=False, na=False), 'Designer',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Dev Evangelist/Advocate'  , case=False, na=False), 'Dev_Evangelist_Advocate',
                           np.where( DF_Saude_Mental['Q62'].str.contains('DevOps/SysAdmin'          , case=False, na=False), 'DevOps_SysAdmin',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Executive Leadership'     , case=False, na=False), 'Executive_Leadership',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Front-end Developer'      , case=False, na=False), 'Front_And_Developer',
                           np.where( DF_Saude_Mental['Q62'].str.contains('HR'                       , case=False, na=False), 'HR',
                           np.where( DF_Saude_Mental['Q62'].str.contains('One-person shop'          , case=False, na=False), 'One_person_shop',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Sale'                     , case=False, na=False), 'Sale',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Supervisor/Team Lead'     , case=False, na=False), 'Supervisor_Team_Lead',
                           np.where( DF_Saude_Mental['Q62'].str.contains('Support'                  , case=False, na=False), 'Support', 
                           np.nan)))))))))))

## Do you work remotely?
DF_Saude_Mental['Q63'].unique()# 

DF_Saude_Mental['Q63'].unique()#'No', 'Yes', 'Maybe'

DF_Saude_Mental['Q63']  =   np.where(np.isin(DF_Saude_Mental['Q63'], [ 'Sometimes']), 'A1',
                            np.where(np.isin(DF_Saude_Mental['Q63'], [ 'Always'])   , 'A2',
                            np.where(np.isin(DF_Saude_Mental['Q63'], [ 'Never'])    , 'A3',
                            np.nan)))

#  What country do you work in?'
DF_Saude_Mental['Q60']  = DF_Saude_Mental['Q60'].str.replace( " ", "_", regex=False)

##DF_Saude_M_Dummie = DF_Saude_M_Dummie.applymap(lambda x: x.replace('.', '_') if isinstance(x, str) else x)

'''
applymap(): Aplica a transformação em todas as células do DataFrame.
lambda x: x.replace('.', '_') if isinstance(x, str) else x:
Se a célula contém texto, substitui . por _.
Se for número, mantém o valor original.

DF_Saude_M_Dummie['Nome_da_Coluna'] = DF_Saude_M_Dummie['Nome_da_Coluna'].str.replace('.', '_', regex=True)

'''


#%%---------------------------------------------
## Análise com algumas variáveis
# ----------------------------------------------

Variaveis2 = ['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q15', 'Q26', 'Q27', 'Q28', 'Q29', 'Q41', 'Q46', 'Q47', 'Q48', 'Q51', 'Q53', 'Q56', 'Q57', 'Q62', 'Q63']
DF_Saude_M2 = DF_Saude_Mental[Variaveis2]


# verificar nulos
DF_Saude_M2.isna().sum()

Explicativa2 = ['Q51']
DF_Y2 = DF_Saude_M2[Explicativa2]
DF_Y2.info()

## Tabela de frequências absolutas da variável: Você foi diagnosticado com um problema de saúde mental por um profissional médico?

DF_Saude_Mental['Q51'].value_counts().sort_index()

#   Q51
#0.0    717
#1.0    716

#DF_X = DF_Saude_M1.drop(columns=['Q51']) # Cria base sem a coluna Y

# ----------------------------------------------
## Criar Dummies
# ----------------------------------------------

DF_Saude_M_Dummie2 = pd.get_dummies(DF_Saude_M2, columns=['Q1', 'Q2', 'Q3', 'Q4', 'Q5', 'Q6', 'Q7', 'Q8', 'Q15', 'Q26', 'Q27', 'Q28', 'Q29', 'Q41', 'Q46', 'Q47', 'Q48', 'Q53', 'Q57', 'Q62', 'Q63'],
                                      dtype=int,
                                      drop_first=True)

DF_Saude_M_Dummie2


 ##(str.replace('[^A-Za-z0-9_]', '_', regex=True)
  DF_Saude_M_Dummie2.columns = DF_Saude_M_Dummie2.columns.str.replace('[^A-Za-z0-9_]', '_', regex=True)

DF_X2 = list(DF_Saude_M_Dummie2.drop(columns=['Q51']).columns)

#%% ESTIMAÇÃO DO MODELO LOGÍSTICO BINÁRIO  

# O argumento 'family=sm.families.Binomial()' da função 'glm' do pacote
#'statsmodels.formula.api' define a estimação de um modelo logístico binário
# Estimação do modelo com as variáveis 'dist', 'sem' e dummies como preditoras


Variaveis_X2= ' + '.join(DF_X2)  # Criar variável q contenha todas as variáveis X - Junta todos os nomes das colunas com '+'
Variaveis_Modelo2 = f'Q51 ~ {Variaveis_X2}' # f',  incorporar expressões e variáveis diretamente dentro de uma strin



Modelo_Logit_SM2= smf.glm(
                    formula=Variaveis_Modelo2, ## variável dependente e as independentes
                    data=DF_Saude_M_Dummie2, ## data: Especifica o DataFrame onde estão as variáveis utilizadas na fórmula.
                    family=sm.families.Binomial() ## family=sm.families.Binomial(): Define que o modelo GLM deve utilizar a família binomial, apropriada para regressão logística.
                    ).fit()

# Parâmetros do 'modelo_logit'
Modelo_Logit_SM2.summary()


Modelo_Logit_SM2 = sm.Logit.from_formula(Variaveis_Modelo2,
                                          DF_Saude_M_Dummie2).fit()

  '''Current function value: 0.260625
  Iterations 8'''

# Parâmetros do 'modelo_challenger'
Modelo_Logit_SM2.summary()

# In[2.4]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests


# Estimação do modelo por meio do procedimento Stepwise
Step_Modelo_Logit_SM2= stepwise(Modelo_Logit_SM2, pvalue_limit=0.05)

#%% Outra forma de aprsentar o output
# 
summary_col([Step_Modelo_Logit_SM2],
            model_names=["Modelo"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs)),
                'Log-lik':lambda x: "{:.3f}".format(x.llf)
        })


#%%
# Calculo dos valores previstos de probabilidade 

DF_Saude_M2['Predct'] = Step_Modelo_Logit_SM2.predict()

DF_Saude_M2


# Construção de função para a definição da matriz de confusão

from sklearn.metrics import confusion_matrix, accuracy_score,\
    ConfusionMatrixDisplay, recall_score

def Matriz_Confusao(predicts, observado, cutoff):
    
    values = predicts.values
    
    predicao_binaria = []
        
    for item in values:
        if item < cutoff:
            predicao_binaria.append(0)
        else:
            predicao_binaria.append(1)
           
    cm = confusion_matrix(predicao_binaria, observado)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.xlabel('True')
    plt.ylabel('Classified')
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.show()
        
    sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
    especificidade = recall_score(observado, predicao_binaria, pos_label=0)
    acuracia = accuracy_score(observado, predicao_binaria)

    # Visualização dos principais indicadores desta matriz de confusão
    indicadores = pd.DataFrame({'Sensitividade':[sensitividade],
                                'Especificidade':[especificidade],
                                'Acurácia':[acuracia]})
    return indicadores

#%% Matrizes de confusão

# Matriz de confusão para cutoff = 0.5

Matriz_Confusao(observado=DF_Saude_M2['Q51'],
                predicts=DF_Saude_M2['Predct'], 
                cutoff=0.5)
'''
   Sensitividade  Especificidade  Acurácia
0       0.891061        0.892608  0.891835

''''


# Igualando critérios de especificidade e de sensitividade

# Criação da função 'espec_sens' para a construção de um dataset com diferentes
#valores de cutoff, sensitividade e especificidade:

def Espec_Sens(observado,predicts):
    
    # Adicionar objeto com os valores dos predicts
    values = predicts.values
    
    # Range dos cutoffs a serem analisados em steps de 0.01
    cutoffs = np.arange(0,1.01,0.10)
    
    # Listas que receberão os resultados de especificidade e sensitividade
    lista_sensitividade = []
    lista_especificidade = []
    
    for cutoff in cutoffs:
        
        predicao_binaria = []
        
        # Definindo resultado binário de acordo com o predict
        for item in values:
            if item >= cutoff:
                predicao_binaria.append(1)
            else:
                predicao_binaria.append(0)
                
        # Cálculo da sensitividade e especificidade no cutoff
        sensitividade = recall_score(observado, predicao_binaria, pos_label=1)
        especificidadee = recall_score(observado, predicao_binaria, pos_label=0)
        
        # Adicionar valores nas listas
        lista_sensitividade.append(sensitividade)
        lista_especificidade.append(especificidadee)
        
    # Criar dataframe com os resultados nos seus respectivos cutoffs
    resultado = pd.DataFrame({'cutoffs':cutoffs,'sensitividade':lista_sensitividade,'especificidade':lista_especificidade})
    return resultado

# In[1.12]: Até o momento, foram extraídos 3 vetores: 'sensitividade',
#'especificidade' e 'cutoffs'. Assim, criamos um dataframe que contém
#os vetores mencionados (dataframe 'dados_plotagem')

Dados_Plotagem = Espec_Sens(observado = DF_Saude_M2['Q51'],
                            predicts = DF_Saude_M2['Predct'])
Dados_Plotagem

# exportar aquivo de cutoffs
Dados_Plotagem.to_excel('C:\\Users\\deniz\\OneDrive\\Documentos\\Python Scripts\\TCC\\Dados_Plotagem_Cutoffs.xlsx', index=False)


# In[1.13]: Plotagem de um gráfico que mostra a variação da especificidade e da
#sensitividade em função do cutoff

plt.figure(figsize=(15,10))
with plt.style.context('seaborn-v0_8-whitegrid'):
    plt.plot(Dados_Plotagem.cutoffs,Dados_Plotagem.sensitividade, marker='o',
         color='indigo', markersize=8)
    plt.plot(Dados_Plotagem.cutoffs,Dados_Plotagem.especificidade, marker='o',
         color='limegreen', markersize=8)
plt.xlabel('Cuttoff', fontsize=20)
plt.ylabel('Sensitividade / Especificidade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.legend(['Sensitividade', 'Especificidade'], fontsize=20)
plt.show()


#%% Construção da curva ROC


fpr, tpr, thresholds =roc_curve(DF_Saude_M2['Q51'], DF_Saude_M2['Predct'])
roc_auc = auc(fpr, tpr)

# Cálculo do coeficiente de GINI
gini = (roc_auc - 0.5)/(0.5)

# Plotando a curva ROC
plt.figure(figsize=(15,10))
plt.plot(fpr, tpr, marker='o', color='darkorchid', markersize=10, linewidth=3)
plt.plot(fpr, fpr, color='gray', linestyle='dashed')
plt.title('Área abaixo da curva: %g' % round(roc_auc, 4) +
          ' | Coeficiente de GINI: %g' % round(gini, 4), fontsize=22)
plt.xlabel('1 - Especificidade', fontsize=20)
plt.ylabel('Sensitividade', fontsize=20)
plt.xticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.yticks(np.arange(0, 1.1, 0.2), fontsize=14)
plt.show()



