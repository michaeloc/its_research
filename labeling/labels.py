import numpy as np
import pandas as pd
from tqdm import tqdm

from preprocess import PreProcess
from preprocess_data import PreprocessData


class Labels(PreprocessData):
    def __init__(self,bus_stop_distance, traffic_light_distance, array_stops,array_trfl):
        super().__init__()
        self.bst_dist = bus_stop_distance
        self.trfl_dist = traffic_light_distance
        self.array_stops = array_stops
        self.array_trfl = array_trfl
        self.prepro = PreProcess()
    
    def add_bus_stop_label(self,data):
        ''' this method is used with multiprocessing
        item[4] is the velocity'''
        chunck = []
        for items in tqdm(data):
            final_item = []
            for item in items:
                for stop in self.array_stops:
#                     dist = self.prepro.distance_in_meters([item[0],item[1]], [stop[4],stop[5]])
                    dist = self.prepro.distance_in_meters([item[0],item[1]], [stop[1],stop[2]])
                    if item[4] < 5 and dist < self.bst_dist:
                        print('bustop')
                        item.append('bus_stop')
                        break
                final_item.append(item)
            chunck.append(final_item)
        return chunck
        
    def add_traffic_light_label(self,data):
        chunck = []
        for items in tqdm(data):
            final_item = []
            for item in items:
                for stop in self.array_trfl:
#                     dist = self.prepro.distance_in_meters([item[0],item[1]], [stop[7],stop[8]])
                    dist = self.prepro.distance_in_meters([item[0],item[1]], [stop[1],stop[2]])
                    if item[4] < 5 and dist < self.trfl_dist and item[10] != 'bus_stop':
                        item[10]='traffic_light'
                        break
                final_item.append(item)
            chunck.append(final_item)
        return chunck

    def add_other_stop_label(self,data):
        for items in tqdm(data):
            for item in items:
                if item[4]<5 and item[10]=='in_route':
                    item[10] = 'other_stop'   
    
    def get_false_labels(self,data,label,min_dist):
        ''' Remove labels other_stop that is between bus or traffic_light'''
        count_b, count_a = [],[]
        for items in tqdm(data):
            for idx in range(len(items)-1):
                if idx > 0 and idx < (len(items)-1):
                    lat_lng_b = [items[idx-1][0],items[idx-1][1]]
                    lat_lng_a = [items[idx+1][0],items[idx+1][1]]
                    lat_lng_c = [items[idx][0],items[idx][1]]
                    if items[idx][16]==label and ((items[idx-1][16]==0.0 or items[idx-1][16]==3.0)\
                    and (items[idx+1][16]==0.0 or items[idx+1][16]==3.0))\
                    and (self.prepro.distance_in_meters(lat_lng_c, lat_lng_b)<min_dist or self.prepro.distance_in_meters(lat_lng_c, lat_lng_a)<min_dist):
                        print(f'before:{items[idx-1][16]}----current:{items[idx][16]}----after:{items[idx+1][16]}')
                        print(f'before:{self.prepro.distance_in_meters(lat_lng_c, lat_lng_b)}----after:{self.prepro.distance_in_meters(lat_lng_c, lat_lng_a)}')
                        count_b.append(self.prepro.distance_in_meters(lat_lng_c, lat_lng_b))
                        count_a.append(self.prepro.distance_in_meters(lat_lng_c, lat_lng_a))
                        items[idx][16]=-1
