#!/usr/bin/env python
# coding: utf-8

# # Projeto da disciplina de Introdução à Ciência de Dados
# 
# Estudante: Fernando de Barros (fbwn@cin.ufpe.br)

# ## 1. Carregamento das Bibliotecas e do Dataset

# In[1]:


import math
import sys  
import datetime
import random
from tqdm import tqdm
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import seaborn as sns
from scipy import stats

# Bokeh
from bokeh.plotting import figure, show, output_notebook
from bokeh.tile_providers import get_provider, Vendors, OSM, CARTODBPOSITRON, WIKIMEDIA

pd.set_option('display.max_columns', None)

## Inicialização de variaveis para o Bokeh
output_notebook()
tile_provider = OSM
colors=['red','blue', 'black', 'yellow']


# ## Dublin Bus GPS sample data from Dublin City Council (Insight Project)
# 
# Bus GPS Data Dublin Bus GPS data across Dublin City, from Dublin **City Council'traffic control**, in csv format. 
# 
# Each datapoint (row in the CSV file) has the following entries:
# 
# - Timestamp micro since 1970 01 01 00:00:00 GMT  
# - Line ID  
# - Direction  
# - Journey Pattern ID  
# - Time Frame (The start date of the production time table - in Dublin the production time table starts at 6am and ends at 3am)  
# - Vehicle Journey ID (A given run on the journey pattern)  
# - Operator (Bus operator, not the driver)  
# - Congestion [0=no,1=yes]  
# - Lon WGS84  
# - Lat WGS84
# - Delay (seconds, negative if bus is ahead of schedule)  
# - Block ID (a section ID of the journey pattern)  
# - Vehicle ID  
# - Stop ID  
# - At Stop [0=no,1=yes]
# 
# Fonte: https://data.gov.ie/dataset/dublin-bus-gps-sample-data-from-dublin-city-council-insight-project
# 

# ### Abre arquivo referente aos dias 10~12 de Janeiro de 2013

# ### Colunas originais do dataset

# In[2]:


features = ['timestamp','line_id','direction','journey_id',
            'time_frame','vehicle_journey_id','operator',
            'congestion','lng','lat','delay','block_id',
            'vehicle_id','stop_id', 'stop']


# In[3]:


df = pd.read_csv(
    'data/siri.20130110.csv.gz', 
    compression='gzip', 
    names=features, 
    header=None)
for i in tqdm(range(20130111,20130112,1)):
    data = pd.read_csv(
        'data/siri.'+str(i)+'.csv.gz', 
        compression='gzip', 
        names=features, 
        header=None)
    df = pd.concat([df,data])


# ### Lista com nomes das colunas que formam a chave da trajetória

# In[4]:


trajetoria = [
    'line_id', 'journey_id', 'time_frame',
    'vehicle_journey_id', 'operator', 'vehicle_id'
             ]


# ## Análise inicial do dataset

# In[5]:


df.head()


# In[6]:


df.tail()


# In[7]:


df.info()


# In[8]:


df.describe()


# Retirar a coluna DIRECTION que não apresenta qualquer informação

# In[9]:


df.drop('direction', axis=1, inplace=True)
'direction' in df.columns


# ### Ajusta tipo das features

# #### Boolean

# In[10]:


df.stop = df.stop.astype('bool')
df.congestion = df.congestion.astype('bool')


# #### Datetime (microseconds - us)

# In[11]:


df['datetime'] = pd.to_datetime(df["timestamp"], unit='us')


# ##### Criação de duas novas features derivadas (day, hour)

# In[12]:


df['hour'] = df["datetime"].dt.hour
df['day'] = df["datetime"].dt.day


# ### Categoricos

# Indicar quais das colunas são categoricas:

# In[13]:


list_cats = ['line_id', 'journey_id' , 'time_frame', 
             'vehicle_journey_id', 'operator', 'block_id', 
             'vehicle_id', 'stop_id']


# In[14]:


for cat in list_cats:
    df[cat] = df[cat].astype('category')


# ### Quantos valores distintos há em cada features que compõe a chave?

# In[15]:


lista_tamanhos = map((lambda x: len(df[x].unique())), trajetoria)
mescla = zip(trajetoria, lista_tamanhos)
print('Descritivo de instâncias únicas para cada atributo da chave:')
tuple(mescla)


# ### Como o 'prodution time' é entre 6am e 3am. Existem rotas fora desta janelas?

# In[16]:


df[(3 < df.hour) & (df.hour < 6)]


# ## Análisar trajetórias

# ### ( ! ) Os blocos abaixo foram retirados pois estava dando erro com a alocação de memória na função groupby.

# Eles vão fazem a análise do número de pontos em cada trajetórias e servem de embasamento para a seleção do número mínimo de 50 pontos para validar uma trajetória.
df.info()df_traj = df.groupby(trajetoria).size().reset_index()
df_traj['points'] = df_traj[0]
df_traj.drop(0, axis=1, inplace=True)df_traj.head()df_traj.points.describe()df_traj.points.plot(kind='box')df_traj.points.plot(kind='hist')df_traj['points_log'] = np.log10(df_traj['points'])
df_traj.plot(y='points_log', kind='hist')df_traj.points_log.describe()
# Observamos uma partição dos número de pontos por trajetória um pouco abaixo de 100 pontos. Mais a frente iremos usar o valor limite de 50 pontos para excluir as trajetórias abaixo deste limiar.

# In[17]:


minimo_pontos = 50


# ## Gerar trajetórias

# Na função que gera trajetórias incluimos algumas features aproveitando o fato de que esta geração de trajetórias percorre o dataset linha a linha, uma vez que precisamos identificar um valor limite entre duas amostras do GPS para poder validar a trajetória. Usamos aqui o valor é de 5 minutos.

# In[18]:


gps = ['lat','lng']

def delta_time(t1, t2) -> float:
    '''
    Retorna diferença temporal em segundos
    Ou np.nan se a diferença temporal para o ponto anterior
    for superior a 5 minutos
    '''
    t1 = pd.to_datetime(t1,unit='us')
    t2 = pd.to_datetime(t2,unit='us')
    time = pd.Timedelta(np.abs(t2 - t1))
    if (time.seconds > 5*60):
        return np.nan
    else:
        return time.seconds

def calc_deltas(frame):
    '''
    Retorna o DF com as colunas de delta[tempo, distancia] preenchidas
    Depende do valor da linha anterior (temporalmente)
    Nas funções MAP são enviados os valores da linha presente e da anterior (shift(1))
    '''
    frame.dist_from_old_point = 0
    frame.time_from_old_point = 0
    delta_d = list(map(
        lambda x, y: geodesic(x,y).meters, frame[gps].values[1:], frame[gps].shift(1).values[1:]))
    delta_t = list(map(
        lambda x, y: delta_time(x,y), frame['timestamp'].values[1:], frame['timestamp'].shift(1).values[1:]))
    if (len(frame[frame.isna()]) == 0):
        print('Não há quaisquer valores NaN')
    frame['dist_from_old_point'] = [0, *delta_d]
    frame['time_from_old_point'] = [0, *delta_t]
    return frame


# Repartição do dataset em uma lista contendo vários dataframes, cada um referente a uma trajaetória 

# No GroupBy a seguir todas as instâncias com uma das features listadas como chave com valor NaN são descartadas.

# In[19]:


# Salvar as descartadas para plotagem ao final do notebook
dfs_menos_q_50 = [x for _,x in df.groupby(trajetoria) if ((len(x) < 50) & (len(x) > 10))]

# Repartição
dfs = [x for _,x in df.groupby(trajetoria) if (len(x) > 50)]

# Seleciona 100 trajetórias aleatórias
random_traj = random.sample(range(0, len(dfs)), 100)
dfs = [dfs[i] for i in random_traj]

# Chamas calc_deltas para cada trajetória
dfs = list(map(calc_deltas,dfs))

# concatena as trajetorias de volta em um DF unico
dfs = pd.concat(dfs)


# ### Criação de novas features

# #### Velocidade e Aceleração

# In[20]:


dfs['velocity'] = dfs.dist_from_old_point / dfs.time_from_old_point
dfs['acc'] = dfs.velocity * (1/dfs.time_from_old_point)
dfs['acc'].fillna(value=0, inplace=True)
dfs['velocity'].fillna(value=0, inplace=True)


# #### ID única para cada trajetória

# In[21]:


dfs[trajetoria] = dfs[trajetoria].astype('string')
dfs['ID'] = dfs[trajetoria].agg('-'.join, axis=1).astype('category').cat.codes


# ### Exclusão de pontos que diferem mais que 5 minutos do prévio

# Verifica também se os pontos restantes (anteriores) tem uma a quantidade mínima de pontos para validar a trajetória

# In[22]:


# IDs com np.nan na coluna time_from_old_point
lista_IDs = dfs[dfs.time_from_old_point.isna()].ID.unique()
print('Pontos a serem analisados:')
print(lista_IDs)
def index2remove(ID):
    cc = dfs[dfs.ID==ID]
    if (len(cc) == 0):
        return
    cc.sort_values('timestamp')
    ts = cc[cc['time_from_old_point'].isna()].timestamp.values[0]
    if (len(cc[cc.timestamp <= ts]) < 50):
        #exclui todo mundo no caso de não ter 50 pontos restantes
        print( f'A trajetória {ID} não tem o mínimos de pontos')
        dfs.drop(cc.index, inplace=True)
        return cc.index
    else:
        # exclui só do ponto problemático em diante
        dfs.drop(cc[cc.timestamp >= ts].index, inplace=True)
        return cc[cc.timestamp >= ts].index
remover = list(map(index2remove,lista_IDs))


# In[23]:


lista_IDs = dfs[dfs.time_from_old_point.isna()].ID.values
lista_IDs


# In[24]:


dfs.head()


# In[25]:


#backup
df = dfs.copy()


# In[26]:


df.tail()

df[df.ID==0][['lat','lng','MF_lat','MF_lng']].head(20)
# Abaixo percebemos valores que não podem pertencer a velocidades reais.

# In[27]:


print( f'Por exemplo o valor máximo de {df.velocity.max()} m/s para velocidade ou {df.acc.max()} m/s² para aceleração')


# In[28]:


new_cols = ['velocity', 'acc', 'time_from_old_point', 'dist_from_old_point',]
            #'MF_velocity', 'MF_acc', 'MF_dist_from_old_point']
df[new_cols].describe()


# In[29]:


df[new_cols].plot(kind='box', figsize=(10,6))


# ## Teste de métodos para detecção de outliers

# ### Isolation Forest sobre a velocidade e a aceleração

# In[30]:


from sklearn.ensemble import IsolationForest


# In[31]:


feats = ['velocity','acc']
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(df[feats])
scores = clf.predict(df[feats])
df['outlier'] = scores


# ### Isolation Forest com normalização dos valores de velocidade e aceleração

# In[32]:


from sklearn.preprocessing import MinMaxScaler


# In[33]:


scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[['velocity','acc']]), columns = ['velocity','acc'])
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(df_normalized)
scores = clf.predict(df_normalized)
df['outlier_norm'] = scores


# In[34]:


df[df.outlier == 1][feats].describe()


# In[35]:


df[df.outlier_norm== 1][feats].describe()


# In[36]:


df[df.outlier == 1][feats].hist()


# Ambos os método retornam valores plausiveis de velocidade e aceleração

# Entretanto, velocidade e aceleração foram valores criados apartir das distâncias entre pontos. 

# ### Poderiamos aplicar a detecção de outliers nestas features geradoras ou o cálculo de velocidade e aceleração nos ajudou a discriminar melhor os outliers?

# ### Isolation Forest a partir das colunas distância e tempo desde o ponto anterior

# In[37]:


feats_0 = ['dist_from_old_point','time_from_old_point']
rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(df[feats_0])
scores = clf.predict(df[feats_0])
df['outlier_0'] = scores


# In[38]:


df[df.outlier_0== 1][feats_0].describe()


# Como os valores não são plausíveis. Concluímos que a geração das novas features: velocidade e aceleração foram significativas para a detecção destes pontos anomalos.

# ### Verificar se o método Local Outlier Factor oferece também uma resposta plausível

# In[39]:


from sklearn.neighbors import LocalOutlierFactor


# In[40]:


clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
scores = clf.fit_predict(df[feats])
df['outlier2'] = scores


# In[41]:


scaler = MinMaxScaler()
df_normalized = pd.DataFrame(scaler.fit_transform(df[['velocity','acc']]), columns = ['velocity','acc'])
rng = np.random.RandomState(42)
clf = LocalOutlierFactor(n_neighbors=10, contamination=0.1)
scores = clf.fit_predict(df_normalized)
df['outlier2_norm'] = scores


# In[42]:


df[df.outlier2 == 1][feats].describe()


# In[43]:


df[df.outlier2_norm == 1][feats].describe()


# Os valores de máximos ainda apresentam pontos que não se adequam à realidade.

# ## Testar detecção de outliers MEAN FILTER

# In[44]:


df['MF_lat'] = df['lat'].rolling(5, min_periods=1).mean()
df['MF_lng'] = df['lng'].rolling(5, min_periods=1).mean()


# In[45]:


gps = ['MF_lat','MF_lng']

def delta_time(t1, t2) -> float:
    '''
    Retorna diferença temporal em segundos
    Ou np.nan se a diferença temporal para o ponto anterior
    for superior a 5 minutos
    '''
    t1 = pd.to_datetime(t1,unit='us')
    t2 = pd.to_datetime(t2,unit='us')
    time = pd.Timedelta(np.abs(t2 - t1))
    if (time.seconds > 5*60):
        return np.nan
    else:        
        return time.seconds

def MF_calc_deltas(frame):
    '''
    Retorna o DF com as colunas de delta[tempo, distancia] preenchidas
    Depende do valor da linha anterior (temporalmente)
    Nas funções MAP são enviados os valores da linha presente e da anterior (shift(1))
    '''
    #frame.dist_from_old_point = 0
    frame.time_from_old_point = 0
    delta_d = list(map(
        lambda x, y: geodesic(x,y).meters, frame[gps].values[1:], frame[gps].shift(1).values[1:]))
    delta_t = list(map(
        lambda x, y: delta_time(x,y), frame['timestamp'].values[1:], frame['timestamp'].shift(1).values[1:]))
    if (len(frame[frame.isna()]) == 0):
        print('Não há quaisquer valores NaN')
    frame['MF_dist_from_old_point'] = [0, *delta_d]
    frame['time_from_old_point'] = [0, *delta_t]
    return frame


# In[46]:


# Repartição
dfs = [x for _,x in df.groupby(trajetoria) if (len(x) > 50)]
# Chamas calc_deltas para cada trajetória
dfs = list(map(MF_calc_deltas,dfs))
# concatena as trajetorias de volta em um DF unico
df = pd.concat(dfs)


# In[47]:


df['MF_vel'] = df['MF_dist_from_old_point'] / df['time_from_old_point']
df['MF_acc'] = df['MF_dist_from_old_point'] / (df['time_from_old_point'] * df['time_from_old_point'])

#df[df.MF_vel == np.inf].time_from_old_point.unique()
df['MF_vel'].fillna(value=0, inplace=True)
df['MF_acc'].fillna(value=0, inplace=True)


# In[48]:


df[['velocity', 'MF_vel']].plot(kind='box')


# In[49]:


feats = ['velocity','acc']
feats_MF = ['MF_vel', 'MF_acc']


# In[50]:


rng = np.random.RandomState(42)
clf = IsolationForest(max_samples=100, random_state=rng)
clf.fit(df[feats_MF])
scores = clf.predict(df[feats_MF])
df['outlier_MF'] = scores


# In[51]:


df[df.outlier== 1][feats].plot(kind='box')


# In[52]:


df[df.outlier_MF== 1][feats_MF].plot(kind='box')


# In[53]:


df.MF_vel.describe()


# In[54]:


df[df.outlier_MF== 1].MF_vel.describe()


# In[55]:


print (f'O uso do Mean Filter sozinho não conseguiu retirar os pontos anômalos do dataset.')
print (f'Apenas após utilização do Mean Filter conjuntamente à detecção com auxílio da técnica Random Forest obtivemos um valor máximo de velocidade plausível.')


# ### Assim iremos excluir quem foi detectado como outlier pelo Random Forest 

# ### Escolhemos o resultado sem a normalização 

# ### Rando Foreste pós normalização e pós Mean Filter também seriam alternativas válidas. Por simplicidade preferimos selecionamos o primeiro método (menos passos).

# In[56]:


df1 = df[df.outlier==1]


# In[57]:


df.dist_from_old_point.unique()


# In[58]:


df1.drop(['outlier', 'outlier_0', 'outlier2','outlier_norm','outlier2_norm','day','stop_id','block_id', 'delay'],axis=1,inplace=True)


# In[59]:


df1.describe()


# In[60]:


df1['acc'].plot(kind='box', figsize=(10,6))


# In[61]:


df1['velocity'].plot(kind='box', figsize=(10,6))


# In[62]:


df1['dist_from_old_point'].plot(kind='box', figsize=(10,6))


# ## Teste de hipótese

# ### Verificação da distribuição da velocidade

# In[63]:


sns.histplot(x=df1.velocity)


# Anderson-Darling test for data coming from a particular distribution.
# 
# The Anderson-Darling test tests the null hypothesis that a sample is
# drawn from a population that follows a particular distribution.
# For the Anderson-Darling test, the critical values depend on
# which distribution is being tested against.  This function works
# for normal, exponential, logistic, or Gumbel (Extreme Value
# Type I) distributions.
# 
# If the returned statistic is larger than these critical values then
# for the corresponding significance level, the null hypothesis that
# the data come from the chosen distribution can be rejected.

# In[64]:


result_test = stats.anderson(df1.velocity, dist='norm')


# In[65]:


print(f'Statistic: {result_test[0]}')

print(f'Critical Value: {result_test[1]}')

print(f'Significance Level: {result_test[2]}')


# In[66]:


print( f'Como o valor {result_test[0]} é maior que qualquer valor crítico {result_test[1]}, então o teste é significativo.')
print('Assim podemos rejeitar a hipótese nula que os pontos vem de uma distribuição normal')


# ### Distribuição do log da velocidade (valores diferentes de zero)

# In[67]:


sns.histplot(x=df1[df1.velocity!=0].velocity.apply(np.log10))


# ### Teste de normalidade para log10(velocidade!=0)

# In[68]:


result_test = stats.anderson(df1[df1.velocity!=0].velocity.apply(np.log10), dist='norm')


# In[69]:


print(f'Statistic: {result_test[0]}')

print(f'Critical Value: {result_test[1]}')

print(f'Significance Level: {result_test[2]}')


# Assim não iremos admitir que haja uma distribuição simétrica na velocidade.

# ### Para quais horários há uma diferença entre a velocidade capturada antes e após?

# ### Mann-Whitney rank test

# A escolha do teste se dá por ausência de normalidade na distribuição e independência entre as trajetórias

# Há um caso limite em que uma mesma trajetória ocorre através de horas diferentes, o que não seria uma medição independente. Porém para cada hora o número de medições em viagens independentes (que acontecem totalmente antes/depois) é muito maior do que as medições em viagens que cruzaram aquele horário.

# In[70]:


hours = df1.hour.unique()
lh_vel = [stats.mannwhitneyu(df1[df1.hour <= h].velocity, df1[df1.hour > h].velocity)[1] for h in hours]


# In[71]:


x = pd.DataFrame(data={'pvalues_vel':lh_vel}, index=hours).sort_index()
x.plot(kind='bar')


# In[72]:


x = x.reset_index()


# In[102]:


x1 = x[x.pvalues_vel<0.05]


# In[105]:


x1.index = x1['index']


# In[107]:


x1.pvalues_vel.plot(kind='bar')


# In[74]:


sns.boxplot(df1.velocity)


# In[75]:


g = sns.FacetGrid(df1[df1.hour.isin(x1['index'].values)], col="hour", col_wrap=4)
g.map(sns.boxplot, 'velocity')


# In[76]:


horas = x1['index'].values


# In[78]:


print( f'Para cada uma das seguintes horas: {horas}')
print('Podemos dizer que há uma diferença entre as velocidades amostradas antes e após a mesma.')
print()
print('Por exemplo:')
print( f'Como o pvalue para as {horas[4]} horas é:\t {x1.iloc[4].pvalues_vel}:')
print( 'Podemos negar H0; Negamos que a distribuiçãos das velocidades sejam iguais.')
print( f'Podemos dizr que há uma difença entre as velocidades antes e após as {horas[4]} horas')


# ### Quais linhas ('line_id') tem uma mediana da velocidade maior/menor que a mediana geral
# 

# #### Wilcoxon signed-rank test (one-side)

# The Wilcoxon signed-rank test tests the null hypothesis that two
# related paired samples come from the same distribution. In particular,
# it tests whether the distribution of the differences x - y is symmetric
# about zero. It is a non-parametric version of the paired T-test.
# 

# The one-sided test has the null hypothesis that the median is 
# positive against the alternative that it is negative 
# (``alternative == 'less'``), or vice versa (``alternative == 'greater.'``).

# alternative : {"two-sided", "greater", "less"}, optional
#     The alternative hypothesis to be tested, see Notes. Default is
#     "two-sided".

# In[79]:


lines = df1.line_id.unique()
gt_vel = [stats.wilcoxon(df1[df1.line_id == t].velocity - df1.velocity.median(), alternative='greater')[1]
          for t in  lines]
lt_vel = [stats.wilcoxon(df1[df1.line_id == t].velocity - df1.velocity.median(), alternative='less')[1]
          for t in  lines]


# In[80]:


dfg = pd.DataFrame(data={'pvalues_vel':gt_vel}, index=lines)
dfl = pd.DataFrame(data={'pvalues_vel':lt_vel}, index=lines)
#diff_median = dfp[(dfp.pvalues_vel < 0.05)].index


# In[81]:


dfg[(dfg.pvalues_vel < 0.05)].plot(kind='bar', figsize=(30,10))


# In[82]:


dfl[(dfl.pvalues_vel < 0.05)].plot(kind='bar', figsize=(30,10))


# In[83]:


print( f'Rejeitamos a hipótese de que estas {len(dfg[(dfg.pvalues_vel < 0.05)])} linhas: \n\n{dfg[(dfg.pvalues_vel < 0.05)].index.values}')
print()
print('(H0) possuem a mediana de suas velocidades menores que a mediana de toda a amostra')
print()
print('Aceitamos a hipótese alternativa que estas linhas tem uma mediana superior à mediana da amostra')


# In[84]:


print( f'Rejeitamos a hipótese de que estas {len(dfl[(dfl.pvalues_vel < 0.05)])} linhas: \n\n{dfl[(dfl.pvalues_vel < 0.05)].index.values}')
print()
print('(H0) possuem a mediana de suas velocidades maiores que a mediana de toda a amostra')
print()
print('Aceitamos a hipótese alternativa que estas linhas tem uma mediana inferior à mediana da amostra')


# ## Plotar algumas trajetórias

# ### Funções auxíliares

# In[85]:


def to_mercator(Coords):
    Coordinates =  Coords
    #literal_eval(Coords)        
    lat = Coordinates[1]
    lon = Coordinates[0]
    
    r_major = 6378137.000
    x = r_major * math.radians(lon)
    scale = x/lon
    y = 180.0/math.pi * math.log(math.tan(math.pi/4.0 + 
        lat * (math.pi/180.0)/2.0)) * scale    
    
    return pd.Series((x, y))

def plot_fig(df_x, color):
    p = figure(x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(get_provider(tile_provider))
    #p.circle(
    p.line(
        x='lng_m',
        y='lat_m',
        color= color,
        source=df_x
    )
    try:
        p.circle(
            x='lng_m',
            y='lat_m',
            color= 'red',
            source=df_x[df_x.outlier == -1]
        )
    except:
        pass
    p.circle(
        x=df_x.iloc[0]['lng_m'],
        y=df_x.iloc[0]['lat_m'],
        color= 'blue',    
    )
    return (p)

def plot_map(data_line):
    p = figure(x_axis_type="mercator", y_axis_type="mercator")
    p.add_tile(get_provider(tile_provider))
    data_line[['lng_m','lat_m']] = data_line[['lng', 'lat']].apply(to_mercator, axis=1)
    p = plot_fig(data_line, 'black')
    show(p)


# ### A linha preta é a trajetória

# ### O ponto azul é a partida

# ### Os pontos vermelhos os outliers

# #### Pontos Excluídos por ter menos que 50 pontos

# In[86]:


plot_map(dfs_menos_q_50[0])
plot_map(dfs_menos_q_50[1])
plot_map(dfs_menos_q_50[10])


# #### Algumas trajetórias válidas

# In[87]:


plot_map(df[df.ID == 10])
plot_map(df[df.ID == 66])
plot_map(df[df.ID == 99])


# #### Alguns pontos que mesmo sendo detectados como válidos não aparentam ser quando plotados podem aparecer.

# Observando a trajetória com maior distancia percorrida podemos ver se há algo equivocado. 
# Podemos incrementar a exclusão de outliers retirando trajetórias muito curta ou muito longas basedas na distância total.

# In[88]:


df1.dist_from_old_point.unique()


# In[89]:


accum = df1.groupby(['line_id', 'ID' ]).dist_from_old_point.sum()
long_run = accum.max()


# In[90]:


accum.reset_index()


# In[91]:


accum.hist()


# In[92]:


accum.plot(kind='box')


# In[93]:


accum = accum.reset_index()
accum.columns = ['line', 'ID', 'total_dist']
accum.line = accum.line.astype('float')


# In[98]:


g = sns.FacetGrid(accum, col="line", col_wrap=4)
g.map(sns.boxplot, 'total_dist')
#sns.boxplot(data=accum, x='total_dist', hue='line')
#("alive", col = "deck", col_wrap = 3,data = df[df.deck.notnull()],kind = "count")
#sns.boxplot(x=var, y='SalePrice', data=housing, ax=subplot)

