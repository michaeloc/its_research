import math
import pandas as pd
import numpy as np
from geopy.distance import geodesic
from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql

class PreProcess():

    def __init__(self, traj_ID: list, coordinates,
                 #types_psql,
                 categories=[], booleans=[], ints=[], drops=[],
                 timestamp='', timestamp_unit='', min_points=50):
        self.traj_ID_list = traj_ID
        self.coordinates = coordinates
        self.min_points = min_points
        self.categories = categories
        self.booleans = booleans
        self.ints = ints
        self.types_psql = dict(zip(traj_ID, types_psql))
        self.table_meta = ''
        self.table_points = ''
        self.timestamp = timestamp
        self.timestamp_unit = timestamp_unit
        self.drops = drops

    def __call__(self, data):
        data = self.clean_data(data)
        data = self.set_types(data)
        meta_table = self._save_trajectories_metadata(data, self.types_psql)
        data = self.add_ID(data, meta_table)
        data = self.set_new_features(data)
        return data

    def _save_trajectories_metadata(self, data):
        self.engine = create_engine('')
        data.to_sql(
            self.table_meta,
            engine,
            if_exists='append',
            index=False,
            dtype=self.types_psql
        )
        self.meta_table = pd.read_sql_table(
            'dublin',
            self.engine
        )
        return meta_table


    "            'line_id' : postgresql.INTEGER,\n",
    "            'journey_id' : postgresql.VARCHAR,\n",
    "            'timeframe' : postgresql.TIMESTAMP,\n",
    "            'vehicle_journey_id' : postgresql.INTEGER,\n",
    "            'operator' : postgresql.VARCHAR,\n",
    "            'vehicle_id' : postgresql.INTEGER,\n",
    "            'trajectory_size' : postgresql.NUMERIC    \n",
