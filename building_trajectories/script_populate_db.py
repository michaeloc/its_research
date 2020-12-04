import pandas as pd
from tqdm import tqdm
from sqlalchemy.dialects import postgresql

import preprocess2 as preproc
from feat_eng import Feat_eng
from config import (engine_local, features, trajetoria, list_ints, PATH_LOCAL,
                    PSQL_TYPES_META_TRAJ, PSQL_TYPE_PT, query_str, query_sql)

engine = engine_local
PATH = PATH_LOCAL
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
    types_psql = dict(zip(traj_ID_list,PSQL_TYPES_META_TRAJ ))
    save_in_db(data, 'dublin', types_psql)

def save_trajectories_points(data):
    data = transform_columns(data)
    psql_types_pt = dict(zip(data.columns, PSQL_TYPE_PT))
    print(psql_types_pt)
    data = data.sort_values('instant')
    save_in_db(data,'dublin_trajectories', psql_types_pt)

def add_ID(data):
    dfs = [x for _,x in data.groupby(trajetoria)]
    dfs = list(map(make_ID,dfs))
    #concatena as trajetorias de volta em um DF unico
    return pd.concat(dfs)

def make_ID(data):
    conn = engine.connect()
    query = query_sql.format(*data[trajetoria].iloc[0])
    query_result = conn.execute( f"SELECT * FROM dublin WHERE {query};").fetchall()
    try:
        data['trajectory_id'] = query_result[0][0]
    except:
        print( f"[!] CAN'T RETURN TRAJECTORY_ID FOR: {query}")
    conn.close()
    return data

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
    return df


def delete_past_db():
    print('Deleting previous databases')
    conn = engine.connect()
    conn.execute('DELETE FROM dublin_trajectories;')
    conn.execute('DELETE FROM dublin;')
    print('Done!')

def open_df(i):


    print(PATH.format('0'+str(i)))
    if (i < 10):
        return pd.read_csv(PATH.format('0'+str(i)), names=features)
    else:
        return pd.read_csv(PATH.format(str(i)), names=features)

if __name__ == '__main__':


    #delete_past_db()

    preprocess = preproc.PreProcess(
        trajetoria, ['lat', 'lng'], ints=list_ints,
        timestamp='timestamp', timestamp_unit='us')

    for i in tqdm(range(1, 31)):
        df = open_df(i)
        df = preprocess(df)
        save_trajectories_metadata(df, trajetoria)
        df = add_ID(df)
        df = features_engineering(df)
        save_trajectories_points(df)
