import pandas as pd
from tqdm import tqdm
from sqlalchemy.dialects import postgresql

import preprocess2 as preproc
import meta_trajectories as meta_traj
from config import engine, features, trajetoria, list_ints, PATH, PSQL_TYPES_META_TRAJ, PSQL_TYPE_PT, query_str

pd.set_option('display.max_columns', None)

def save_in_db(data, name_table, types_psql):
    return data.to_sql(
        name_table,
        engine,
        if_exists='append',
        index=False,
        dtype=types_psql
    )

def save_trajectories_metadata(data, traj_ID_list):
    data = data.groupby(traj_ID_list).size().reset_index()
    data = data[traj_ID_list]
    save_in_db(data, 'dublin', PSQL_TYPES_META_TRAJ)

def save_trajectories_points(data):
    data = transform_columns(data)
    psql_types_pt = dict(zip(data.columns, PSQL_TYPE_PT))
    save_in_db(df,'dublin_trajectories', psql_types_pt)

def add_ID(self, data):
    data = [x for _,x in data.groupby(self.traj_ID_list)]
    data = list(map(make_ID,data))
    #concatena as trajetorias de volta em um DF unico
    return pd.concat(data)

def make_ID(self, data):
    query = query_str.format(*data[self.traj_ID_list].iloc[0])
    print(query)
    data['trajectory_id'] = self.meta_table.query(query).trajectory_id.iloc[0]
    return data

def open_df(i):
    return pd.read_csv(PATH.format('0'+str(i)), names=features) if (i < 10) \
        else pd.read_csv(PATH.format(str(i)), names=features)

def transform_columns(data):
    data = data[['trajectory_id', '_timestamp', 'lat', 'lng', 'lat_mercator',
                 'lng_mercator', 'speed', 'acceleration','dist_old_point',
                 'time_old_point','bearing', 'year', 'month', 'day', 'hour',
                 'minute']]
    data.columns = ['trajectory_id', 'instant', 'lat', 'lng',
                  'lat_mercator', 'lng_mercator', 'speed', 'acceleration',
                  'delta_dist', 'delta_time', 'bearing', 'year',
                  'month', 'day', 'hour', 'miniute']
    return data

def features_engineering(df):
    feat = Feat_eng(['lat', 'lng'], timestamp='timestamp')
    df = feat(df)

if __name__ == '__main__':
    preprocess = preproc.PreProcess(
        trajetoria, ['lat', 'lng'], TYPES_PSQL, ints=list_ints,
        timestamp='timestamp', timestamp_unit='us')

    for i in tqdm(range(1, 31)):
        df= open_df(i)
        df = preprocess(df)
        save_trajectories_metadata(df, trajetoria)
        df = add_ID(df, trajetoria)
        df = features_engineering(df)
        save_trajectories_points(df)

