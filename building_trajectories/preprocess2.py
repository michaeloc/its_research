import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances
import pandas as pd
import copy
import numpy as np
import utm
import math
import geopy.distance
import uuid

class PreProcess():

    def __init__(self, traj_ID: list, coordinates, min_points=50,
                 categories=[], booleans=[], timestamp='',
                 timestamp_unit='', drops=[]):
        self.traj_ID_list = traj_ID
        self.coordinates = coordinates
        self.min_points = min_points
        self.categories = categories
        self.booleans = booleans
        self.timestamp = timestamp
        self.timestamp_type = timestamp_type
        self.drops = drops

    def __call__(self, data):
        data = self.clean_data()
        data = self.make_ID(data)
        data = self.set_types(data)
        data = self.set_new_features(data)
        return data

    def clean_data(self, data):
        data.drop(self.drops, axis=1, inplace=True)
        ##retirar linhas com NAN nos campos da chave
        return data

    def make_ID(self, data):
        if (len(traj_ID_list)==1):
            self.traj_ID = traj_ID_list[0]
        else:
            data_tmp = data[trajetoria].astype('string')
            data['ID'] = data_tmp.agg('-'.join, axis=1).astype('category').cat.codes
        return data

    def set_types(self, data):
        data[self.categories] = data[self.categories].astype('category')
        data[self.booleans] = data[self.booleans].astype('bool')
        data[ f'_{self.timestamp}'] = pd.to_datetime(df.timestamp, unit=self.timestamp_unit)
        return data


    def set_new_features(self, data):
        data['year']= data[ f'_{self.timestamp}'].dt.year
        data['month']= data[ f'_{self.timestamp}'].dt.month
        data['day']= data[ f'_{self.timestamp}'].dt.day
        data['hour']= data[ f'_{self.timestamp}'].dt.hour
        data['minute']= data[ f'_{self.timestamp}'].dt.minute
        data = self._add_deltas(data)
        data = self._add_speed(data)
        data = self._add_acceleration(data)
        data = self.create_mercator_coord(data)
        data = _discretize_time(data)
        return data

    def _add_speed(self, data):
        data['speed'] = data.dist_old_point / data.time_old_point
        ## Se a divisão foi por zero (resulta NaN) é pq o gps pegou dois ponto
        ## com mesmo timestamp, assim atribui velocidade zero
        ## pq deslocamento zero tmb - Verificar se não distorce os dados!
        data['speed'].fillna(value=0, inplace=True)
        return data

    def _add_acceleration(self, data):
        data['acceleration'] = data.dist_old_point / (data.time_old_point * data.time_old_point)
        ## Se a divisão foi por zero (resulta NaN) é pq o gps pegou dois ponto
        ## com mesmo timestamp, assim atribui aceleração zero
        ## pq deslocamento zero tmb - Verificar se não distorce os dados!
        data['acceleration'].fillna(value=0, inplace=True)
        return data

    def _add_deltas(self, data):
        '''
        Add deltas time/space
        '''
        data = [x for _,x in data.groupby(self.traj_ID) if (len(x) > self.min_points)]
        # Chamas calc_deltas para cada trajetória
        data = list(map(self._calc_deltas,data))
        #concatena as trajetorias de volta em um DF unico
        return pd.concat(data)

    @staticmethod
    def _delta_time(t1, t2):
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

    def _calc_deltas(self, data_trajetoria):
        '''
        Retorna o DF com as colunas de delta[tempo, distancia] preenchidas
        Depende do valor da linha anterior (temporalmente)
        Nas funções MAP são enviados os valores da linha presente e da anterior (shift(1))
        '''
        data_trajetoria['dist_old_point'] = 0
        data_trajetoria['time_old_point'] = 0
        delta_d = list(map(
            lambda x, y: geodesic(x,y).meters,
            data_trajetoria[self.coordinates].values[1:],
            data_trajetoria[self.coordinates].shift(1).values[1:]
        ))
        delta_t = list(map(
            lambda x, y: _delta_time(x,y),
            data_trajetoria[self.timestamp].values[1:],
            data_trajetoria[self.timestamp].shift(1).values[1:]
        ))
        data_trajetoria['dist_old_point'] = [0, *delta_d]
        data_trajetoria['time_old_point'] = [0, *delta_t]
        data_trajetoria = self._remove_deltatime_gt_5min(self, data_trajetoria)
        return data_trajetoria

    def _remove_deltatime_gt_5min(self, trajetoria):
        trajetoria.sort_values('timestamp')
        ## Só pega o primeiro gap de 5 minutos
        ts = trajetoria[trajetoria['time_old_point'].isna()][self.timestamp].values[0]
        if (len(trajetoria[trajetoria.self.timestamp <= ts]) < 50):
            #exclui todo mundo no caso de não ter 50 pontos restantes
            return None
        else:
            # exclui só do ponto problemático em diante
            return trajetoria[trajetoria.self.timestamp < ts].index

    def create_mercator_coord(self, data):
        data[['lat_mercator','lon_mercator']] = data[self.coordinates].apply(_to_mercator, axis=1)
        return data

    @staticmethod
    def _to_mercator(coordinates):
        lon = coordinates[0]
        lat = coordinates[1]
        r_major = 6378137.000
        x = r_major * math.radians(lon)
        scale = x/lon
        log = math.log(math.tan(math.pi/4.0 +lat * (math.pi/180.0)/2.0))
        y = (180.0/math.pi) * log  * scale
        return pd.Series((x, y))

    @staticmethod
    def _discretize_time(self,data):
        data['day_moment'] = ''
        conditions = [
            not (data['hour'] < 12),
            not (data['hour'] >= 12) & (data['hour'] < 18),
            not (data['hour'] >= 18)
        ]
        data['day_moment'].where(conditions[0], 'MORNING',inplace=true)
        data['day_moment'].where(conditions[1], 'AFTERNOON' ,inplace=true)
        data['day_moment'].where(conditions[2], 'NIGHT',inplace=true)
        return data



def main():
    df = pd.read_csv('../data/siri.20130101.csv.gz')
    preprocess = PreProcess()
    df = preprocess(df)

if __name__ == '__main__':
    #main()

