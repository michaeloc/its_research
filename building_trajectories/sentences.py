from preprocess_data import PreprocessData
from tqdm import tqdm
from preprocess import PreProcess
import datetime
import numpy as np
import pandas as pd
import geopy
from math import degrees
import math
import copy
'''
Features to use this class
['lat','lng','instante','rota','velocidade','posicao','viaje','matricula_id','lat_uber','lng_uber','label']
'''
class Sentences(PreprocessData):

    _cached_data = None

    def __init__(self, data, trajectory_features):
        super().__init__()
        self.full_features = data.columns
        self.trajectory_features = (trajectory_features)
        self.prepro = PreProcess()

    def _has_min_quantity_of_points(self, items):
        return len(items) > 10

    def is_window(self, delta_time):
        return delta_time < 5 * 60

    def delta_time(self, t1, t2) -> float:
        ##Return time difference between time in seconds
        t1 = pd.to_datetime(t1,unit='us')
        t2 = pd.to_datetime(t2,unit='us')
        delta = pd.Timedelta(np.abs(t2 - t1))
        return delta.seconds

    def create_sentences(self, data) -> list:
        data = data.sort_values('timestamp')
        actual_trajectory = data[self.trajectory_features].iloc[0].values
        actual_time = data.iloc[0].timestamp
        partial_list, complete_trajectory = [], []
        iterator = 0
        for index in tqdm(data.index):
            if (self.is_valid_point(data, index, actual_trajectory)):
                delta = self.delta_time(actual_time, data.loc[index].timestamp)
                if self.is_window(delta):
                    partial_list.append(data.loc[index].tolist())
            actual_time = data.loc[index].timestamp
            iterator += 1
        if self._has_min_quantity_of_points(partial_list):
            complete_trajectory.append(partial_list)
        return complete_trajectory

    def is_valid_point(self, data, idx, actual_trajectory):
        return all(data[self.trajectory_features].loc[idx] == actual_trajectory)

    def label_encoder(self, data):
        for items in tqdm(data):
            for item in items:
                if item[10] == 'bus_stop':
                    item[10] = 0.0
                elif item[10] == 'in_route':
                    item[10] = 1.0
                elif item[10] == 'other_stop':
                    item[10] = 2.0
                else:
                    item[10] = 3.0

    def bearing(self, point1, point2):
        lat1 = math.radians(point1[0])

        lat2 = math.radians(point2[0])

        y = math.sin(math.radians(point2[1] - point1[1])) * math.cos(lat2)

        x = math.cos(lat1) * math.sin(lat2) - (
            math.sin(lat1) * math.cos(lat2) *
            math.cos(math.radians(point2[1] - point1[1])))

        deg = degrees(math.atan2(y, x))
        return (deg + 360) % 360

    def acceleration(self, v1, v2, deltaT) -> float:
        #         v1 e v2 devems ser m/s
        return np.abs(v1 - v2) / deltaT

    def velocity(self, deltaT, deltaS) -> float:
        return deltaS / deltaT

    def delta_space(self, s1, s2) -> float:
        return self.prepro.distance_in_meters(s1, s2)

    def get_frmt(self, date):
        return '%Y-%m-%d %H:%M:%S.%f' if len(
            date) > 19 else '%Y-%m-%d %H:%M:%S'

    def days_of_week(self, t1):
        f1 = self.get_frmt(t1)
        t1 = datetime.datetime.strptime(t1, f1)
        return float(t1.weekday())

    def hours_of_day(self, t1):
        f1 = self.get_frmt(t1)
        t1 = datetime.datetime.strptime(t1, f1)
        return float(t1.hour)

    def complete_trajectory(self, item, pad):
        new_trajectory = list()
        diff = abs(pad - len(item))
        if len(item) > pad:
            new_trajectory = item[:pad]
            return new_trajectory
        elif len(item) < pad:
            new_trajectory = item
            new_trajectory.extend([item[len(item) - 1]] * diff)
            return new_trajectory
        return item

    def get_time_in_seconds(self, data):
        ## returns values in seconds
        for items in tqdm(data):
            for idx, item in enumerate(items):
                if type(item[2]) == str:
                    frmt = self.get_frmt(item[2])
                    date_2 = datetime.datetime.strptime(item[2], frmt)
                    item[2] = date_2.timestamp()
        return data

    def put_statistics_metrics(self, data, window=16):
        # It takes windows and calculates statistics
        final_list_x_b, final_list_x_a, final_list_x_c, final_list_x_as, final_list_x_bs = list(
        ), list(), list(), list(), list()
        final_list_y = list()
        final_list_ys = list()
        features = [4, 5, 6, 7, 11]
        basic_features = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]
        basic_features_c = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11, 3, 12, 13, 15]
        for item in tqdm(data):
            for i in range(0, len(item), 1):
                if i >= window and i + window <= len(
                        item) - 1 and item[i][14] != -1 and item[i][14] != 1.0:
                    before = item[abs(i - window):i][:, features]
                    after = item[i + 1:i + window + 1][:, features]

                    mean_before = np.mean(before, axis=0)
                    mean_after = np.mean(after, axis=0)
                    std_before = np.std(before, axis=0)
                    std_after = np.std(after, axis=0)
                    min_before = np.min(before, axis=0)
                    min_after = np.min(after, axis=0)
                    max_before = np.max(before, axis=0)
                    max_after = np.max(after, axis=0)
                    median_before = np.median(before, axis=0)
                    median_after = np.median(after, axis=0)
                    before = np.concatenate(
                        (mean_before, std_before, min_before, max_before,
                         median_before)).tolist()
                    after = np.concatenate((mean_after, std_after, min_after,
                                            max_after, median_after)).tolist()

                    final_list_x_b.append(
                        item[abs(i - window):i][:, basic_features])
                    final_list_x_a.append(item[i + 1:i + window +
                                               1][:, basic_features])
                    final_list_x_c.append(item[i, basic_features_c])
                    final_list_x_bs.append(before)
                    final_list_x_as.append(after)

                    final_list_y.append(item[i][14])

                    final_list_ys.append(
                        np.array((item[abs(i - window):i, 14].tolist() +
                                  [item[i][14]] +
                                  item[i + 1:i + window + 1, 14].tolist())))

        return final_list_x_b, final_list_x_a, final_list_x_c, final_list_x_bs, final_list_x_as, final_list_y, final_list_ys

    def put_statistics_metrics_with_padding(self, data, window=16):
        # It takes windows and calculates statistics
        # 16 é flag que informa o ruído
        final_list_x_b, final_list_x_a, final_list_x_c, final_list_x_as, final_list_x_bs = list(
        ), list(), list(), list(), list()
        final_list_y = list()
        final_list_ys = list()
        features = [4, 5, 6, 7, 11]
        basic_features = [0, 1, 2, 4, 5, 6, 7, 9, 10, 11]
        basic_features_c = [
            0, 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 3, 12, 13, 15, 16, 17
        ]
        for item in tqdm(data):
            for i in range(0, len(item), 1):
                # and item[i][14] != 1.0:
                if i >= window and i + window <= len(
                        item) - 1 and item[i][14] != -1:
                    before = item[abs(i - window):i][:, features]
                    after = item[i + 1:i + window + 1][:, features]

                    mean_before = np.mean(before, axis=0)
                    mean_after = np.mean(after, axis=0)
                    std_before = np.std(before, axis=0)
                    std_after = np.std(after, axis=0)
                    min_before = np.min(before, axis=0)
                    min_after = np.min(after, axis=0)
                    max_before = np.max(before, axis=0)
                    max_after = np.max(after, axis=0)
                    median_before = np.median(before, axis=0)
                    median_after = np.median(after, axis=0)
                    before = np.concatenate(
                        (mean_before, std_before, min_before, max_before,
                         median_before)).tolist()
                    after = np.concatenate((mean_after, std_after, min_after,
                                            max_after, median_after)).tolist()

                    final_list_x_b.append(
                        item[abs(i - window):i][:, basic_features])
                    final_list_x_a.append(item[i + 1:i + window +
                                               1][:, basic_features])
                    final_list_x_c.append(item[i, basic_features_c])
                    final_list_x_bs.append(before)
                    final_list_x_as.append(after)

                    final_list_y.append(item[i][14])

                    final_list_ys.append(
                        np.array((item[abs(i - window):i, 14].tolist() +
                                  [item[i][14]] +
                                  item[i + 1:i + window + 1, 14].tolist())))
        return final_list_x_b, final_list_x_a, final_list_x_c, final_list_x_bs, final_list_x_as, final_list_y, final_list_ys

    def add_features(self, data):
        for items in tqdm(data):
            for idx in range(len(items)):
                if len(items[idx]) <= 11:
                    if idx == 0:
                        items[idx][4] = items[idx][4] / 3.6
                        items[idx].insert(5, 0.0)
                        items[idx].insert(6, 0.0)
                        items[idx].insert(7, 0.0)
                        items[idx].insert(8, 0.0)
                        items[idx].insert(9, self.days_of_week(items[idx][2]))
                        items[idx].insert(10, self.hours_of_day(items[idx][2]))
                    else:
                        v1 = items[idx - 1][4]
                        v2 = items[idx][4] / 3.6
                        p1 = items[idx - 1][:2]
                        p2 = items[idx][:2]
                        t1 = items[idx - 1][2]
                        t2 = items[idx][2]
                        time = self.delta_time(t1, t2)
                        space = self.delta_space(p1, p2)
                        if time == 0:
                            time = 0.00000001
                        #Uso aqui para calcular a nova velocidade e aceleração devido ao ruído espacial
                        vel = self.velocity(time, space)
                        acc = self.acceleration(v1, vel, time)
                        #                 acc = sentences.acceleration(v1,v2,time)
                        bear = self.bearing(p1, p2)
                        #Mudo de v2 para vel por motivo do ruído espacial
                        items[idx][4] = vel
                        items[idx].insert(5, acc)
                        items[idx].insert(6, space)
                        items[idx].insert(7, np.abs(bear - items[idx - 1][7]))
                        items[idx].insert(8, time)
                        items[idx].insert(9, self.days_of_week(t2))
                        items[idx].insert(10, self.hours_of_day(t2))
                        if items[idx][4] * 3.6 > 5 and items[idx][-1] != 1.0:
                            items[idx][-1] = 1.0
                        if items[idx][4] * 3.6 < 5 and items[idx][-1] == 1.0:
                            items[idx][-1] = 2.0

    def select_features(self, data):
        '''
            Select only important features, here we remove 13o fearure and add id point and id   trajectory. Both ids is useful to rebuild the trajectories
        '''
        final_list = list()
        idx = 0
        for i, items in tqdm(enumerate(data)):
            list_item = list()
            for j, item in enumerate(items):
                aux = list()
                aux = copy.copy(item[:12])
                # aqui
                aux.insert(12, item[14])
                aux.insert(13, item[15])
                aux.insert(14, item[16])
                '''adding id in each point of trajectory'''
                aux.insert(15, idx)
                '''adding id to identify each trajectory'''
                aux.insert(16, i)

                idx += 1

                list_item.append(aux)
            final_list.append(list_item)
        return final_list

    def add_id_noise(self, data, data_with_noise):
        '''
        Here, we need pass the index from trajectories with noise, ex: set(np.load('models/id_point_trajectory_without_noise_dublin_clean.npy'))
        '''
        for i, items in tqdm(enumerate(data)):
            if i in data_with_noise:
                for item in items:
                    item.append(1)
            else:
                for item in items:
                    item.append(0)

    def padding(self, pad, data):
        '''
        Ex: padding(16,np.array(final_list_with_time))
        '''
        final_list = list()
        for items in data:
            item_list = list()
            item_list.extend([np.zeros_like(items[0]).tolist()] * pad)
            for item in items:
                item_list.append(item.tolist())
            for i in range(pad):
                item_list.append(items[len(items) - 1].tolist())
            final_list.append(item_list)
        return final_list
