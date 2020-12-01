import math
import pandas as pd
import numpy as np
from geopy.distance import geodesic

class Feat_eng():

    def __init__(self, coordinates, timestamp=''):
        self.coordinates = coordinates
        self.timestamp = timestamp

    def __call__(self, data):
        print('Setting new features')
        data['year']= data[ f'_{self.timestamp}'].dt.year
        data['month']= data[ f'_{self.timestamp}'].dt.month
        data['day']= data[ f'_{self.timestamp}'].dt.day
        data['hour']= data[ f'_{self.timestamp}'].dt.hour
        data['minute']= data[ f'_{self.timestamp}'].dt.minute
        data = self._visit_old_point(data)
        data = self._add_speed(data)
        data = self._add_acceleration(data)
        data = self.create_mercator_coord(data)
        #data = self._discretize_time(data)
        print('Listo!')
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

    def _divide_to_map(self, data, func):
        data = data.sort_values('timestamp')
        data = [x for _,x in data.groupby('trajectory_id'_list)]
        data = list(map(func,data))
        #concatena as trajetorias de volta em um DF unico
        return pd.concat(data)

    def _visit_old_point(self, data):
        data = self._divide_to_map(data,self._calc_deltas)
        data = self._divide_to_map(data,self._calc_bearing)
        return data

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
            lambda x, y: self._delta_time(x,y),
            data_trajetoria[self.timestamp].values[1:],
            data_trajetoria[self.timestamp].shift(1).values[1:]
        ))
        data_trajetoria['dist_old_point'] = [0, *delta_d]
        data_trajetoria['time_old_point'] = [0, *delta_t]
        data_trajetoria = self._remove_deltatime_gt_5min(data_trajetoria)
        return data_trajetoria

    def _calc_bearing(self, data_trajetoria):
        bearing = list(map(
            lambda x, y: self._bearing(x,y),
            data_trajetoria[self.coordinates].shift(1).values[1:],
            data_trajetoria[self.coordinates].values[1:],
        ))
        data_trajetoria['bearing'] = [*bearing, np.nan]
        return data_trajetoria

    def _bearing(self, point1, point2):
        lat1 = math.radians(point1[0])

        lat2 = math.radians(point2[0])

        y = math.sin(math.radians(point2[1] - point1[1])) * math.cos(lat2)

        x = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) *
            math.cos(math.radians(point2[1] - point1[1])))

        deg = math.degrees(math.atan2(x, y))
        return (deg + 360) % 360

    def _remove_deltatime_gt_5min(self, trajetoria):
        ## Só pega o primeiro gap de 5 minutos
        gt5min2oldpoint = trajetoria[trajetoria['time_old_point'].isna()]
        if (len(gt5min2oldpoint) == 0):
            return trajetoria
        ts = gt5min2oldpoint[self.timestamp].values[0]
        if (len(trajetoria[trajetoria[self.timestamp] <= ts]) < 50):
            #exclui todo mundo no caso de não ter 50 pontos restantes
            return None
        else:
            # exclui só do ponto problemático em diante
            return trajetoria[trajetoria[self.timestamp] < ts]

    def create_mercator_coord(self, data):
        data[['lat_mercator','lng_mercator']] = data[self.coordinates].apply(self._to_mercator, axis=1)
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
    def _discretize_time(data):
        data['day_moment'] = ''
        conditions = [
            (not (data['hour'] < 12)),
            (not ((data['hour'] >= 12) & (data['hour'] < 18))),
            (not (data['hour'] >= 18))
        ]
        data['day_moment'].where(conditions[0], 'MORNING',inplace=true)
        data['day_moment'].where(conditions[1], 'AFTERNOON' ,inplace=true)
        data['day_moment'].where(conditions[2], 'NIGHT',inplace=true)
        return data
