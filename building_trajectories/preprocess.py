import pandas as pd
from sklearn.metrics.pairwise import pairwise_distances

class PreProcess():
    import pandas as pd
    import copy
    import numpy as np
    import utm
    import math
    from gmplot import gmplot
    import geopy.distance
    import uuid
    
    from geopy.distance import vincenty

    def __init__(self, treshold=50):
        self.treshold = treshold
    
    def clean_data(self, data):
        non_zero_data = data.where((data['CoordX']!=0) & (data['CoordY']!=0) & (data['Linea']!=0) & (data['Ruta']!=0))
        data_without_nan = non_zero_data.dropna()    
        filter_regex_matricula = data_without_nan.matricula.str.match('[A-Z]{3}[0-9]{4}')  
        return data_without_nan[filter_regex_matricula]   
    
    def select_by_parameter(self,data, parameter, value):
        return data.loc[data[parameter]==value]
    
    def coordinates_to_latlng(self, data):
        lat = []
        lng = []
        pd.set_option("display.precision", 20)
        for row in range(len(data)):
            dt = self.utm.to_latlon(data.iloc[row]['CoordX'],data.iloc[row]['CoordY'],25,'L')
            lat.append(dt[0])
            lng.append(dt[1])
        data['lat'] = self.np.array(lat)
        data['lng'] = self.np.array(lng)
        return data
    
    def discretize_by_time(self,data):
        data = self._object_to_datetime(data)
        
        new_time_discretized = []
        for row in range(len(data)):
            if data.iloc[row]['Instante'].time() < self.pd.to_datetime('11:59:59').time():
                new_time_discretized.append(TimeEnum.MORNING.value)
                print (TimeEnum.MORNING.value)
            elif data.iloc[row]['Instante'].time() > self.pd.to_datetime('11:59:59').time() and data.iloc[row]['Instante'].time() < self.pd.to_datetime('17:59:59').time():
                new_time_discretized.append(TimeEnum.AFTERNOON.value)
                print (TimeEnum.AFTERNOON.value)
            else:
                new_time_discretized.append(TimeEnum.NIGHT.value)
                print(data.iloc[row]['Instante'])
                print(TimeEnum.NIGHT.value)
        data['DiscretizedInstante'] = self.np.array(new_time_discretized)
        return data
    
    def _object_to_datetime(self,data):
        data.loc[:,'Instante'] = self.pd.to_datetime(data['Instante'])
    
    def out_of_bound(self,coord1, coord2):
        return self.geopy.distance.vincenty(coord1,coord2).m > self.treshold

    def cluster_points(self,data):
        similar_idex = {}
        i = 0
        while i < len(data):
            # print('before valor i:{}'.format(i))
            similar_idex[i]=[]
                
            coord_i = (data.iloc[i]['lat'], data.iloc[i]['lng'])
            similar_idex[i].append(data.index[i])
            second_idex = []
            for j in range(i+1,len(data),1):
                coord_j = (data.iloc[j]['lat'], data.iloc[j]['lng'])
                if not self.out_of_bound(coord_i, coord_j):
                    # print('próximo')
                    similar_idex[i].append(data.index[j])
                    second_idex.append(j)
                    # flag = False
                else:
                    flag = False
                    aux_similar_idex = self.copy.copy(similar_idex[i])
                    third_index = []
                    idx_w = 0
                    for w in range(1,len(similar_idex[i]),1):
                        if not self.out_of_bound((data.loc[similar_idex[i][w]]['lat'],data.loc[similar_idex[i][w]]['lng']),(data.iloc[j]['lat'],data.iloc[j]['lng'])):
                            idx_w = second_idex[w-1]
                            flag = True
                        else:
                            flag = False
                            break
                    if flag:
                        similar_idex[i].append(data.index[j])
                        coord_i = (data.loc[idx_w]['lat'],data.loc[idx_w]['lng'])
                        second_idex.append(j)
                    else:
                        i = j
                        break
            if j >= len(data)-1:
                break
            # print('valor i:{}'.format(i))
        return similar_idex 

    def set_id(self,dataframe,clusters):
        dataframe['cluster_id'] = self.np.zeros(len(dataframe))
        for key, values in clusters.items():
            id = str(self.uuid.uuid4())
            for idx in values:
                dataframe.iloc[idx,dataframe.shape[1]-1] = id
        return dataframe
    
    def distance_in_meters(self,x,y):
        return self.vincenty((x[0],x[1]),(y[0],y[1])).m
    
    def calculate_distance_matrix(self,data):
        return pairwise_distances(data, metric=self.distance_in_meters)
    
    def put_labels(self,unique, prototipos, datas):
        data_frame = self.pd.DataFrame()
        for i in unique:
            members = prototipos == i
            values = datas[members]
    #         print(values)
            cls = self.np.array([i for j in range(len(values))])
            values = self.np.concatenate((values,cls.reshape(len(cls),1)), axis=1)
            if len(data_frame)>0:
                data_frame = self.pd.concat([data_frame, self.pd.DataFrame(values,columns=['lat','lng','class'])])
    #             print (data_frame)
    #             break
            else:
                data_frame = self.pd.DataFrame(values,columns=['lat','lng','class'])
    #         break
    #         print(data_frame)
        return data_frame

import enum
class TimeEnum(enum.Enum):
    MORNING = 'morning'
    AFTERNOON = 'afternoon'
    NIGHT = 'night'




def main():
    data_frame = pd.read_csv('data-1535649343264.csv',sep=',')
    preprocess = PreProcess()
    # data_filter = preprocess.clean_data(data_frame)
    # teste = preprocess.select_by_parameter(data_filter,'matricula','KFG9789')
    # lat_lng = preprocess.coordinates_to_latlng(teste)
    # clusters = preprocess.cluster_points(data_frame)
    # print(preprocess.set_id(data_frame,clusters))
    # preprocess.calculate_distance_matrix(data_frame.values[:,1:3])
if __name__ == '__main__':
    main()

