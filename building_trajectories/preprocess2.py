import pandas as pd

class PreProcess():

    def __init__(self, traj_ID: list, categories=[],booleans=[], ints=[],
                 drops=[], timestamp='', timestamp_unit='', min_points=50):
        self.traj_ID_list = traj_ID
        self.min_points = min_points
        self.categories = categories
        self.booleans = booleans
        self.ints = ints
        self.timestamp = timestamp
        self.timestamp_unit = timestamp_unit
        self.drops = drops

    def __call__(self, data):
        print('Cleaning Data')
        data = self.clean_data(data)
        print('Setting types')
        data = self.set_types(data)
        print('Preprocessing Finished!')
        return data

    def clean_data(self, data):
        data.drop(self.drops, axis=1, inplace=True)
        ## Retirar linhas com NAN nos campos da chave
        data.dropna(subset=self.traj_ID_list ,inplace=True)
        ### Retira conjuntos de pontos (trajetórias) com menos de 50 pontos
        data = [x for _,x in data.groupby(self.traj_ID_list) if (len(x) > self.min_points)]
        data = pd.concat(data)
        return data

    def set_types(self, data):
        data[self.categories] = data[self.categories].astype('category')
        data[self.booleans] = data[self.booleans].astype('bool')
        data[self.ints] = data[self.ints].astype('int')
        data[ f'_{self.timestamp}'] = pd.to_datetime(data.timestamp, unit=self.timestamp_unit)
        return data
