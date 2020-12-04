from sqlalchemy import create_engine
from sqlalchemy.dialects import postgresql
from secrets import psql_passwd

engine = create_engine( f'postgresql://postgres:{psql_passwd}@localhost/urbanmobility')
engine_local = create_engine('postgresql://saci@localhost/zeroUm')

query_str = "line_id == {} & journey_id == '{}' &  time_frame == '{}' & vehicle_journey_id == {} & operator == '{}' & vehicle_id == {}"
query_sql = "line_id = {} and journey_id = '{}' and  time_frame = '{}' and vehicle_journey_id = {} and operator = '{}' and vehicle_id = {}"

features = [
    'timestamp', 'line_id', 'direction', 'journey_id',
    'time_frame', 'vehicle_journey_id', 'operator',
    'congestion', 'lng', 'lat', 'delay', 'block_id',
    'vehicle_id', 'stop_id', 'stop']

trajetoria = [
    'line_id', 'journey_id', 'time_frame',
    'vehicle_journey_id', 'operator', 'vehicle_id']

list_ints = ['line_id']

PATH = '../../../segmentation/mobility_its/dublin/siri.201301{}.csv.gz'
PATH_LOCAL = '../data/siri.201301{}.csv.gz'

PSQL_TYPES_META_TRAJ = {
    postgresql.INTEGER,
    postgresql.VARCHAR,
    postgresql.TIMESTAMP,
    postgresql.INTEGER,
    postgresql.VARCHAR,
    postgresql.INTEGER,
    # 'trajectory_size' : postgresql.NUMERIC·
}

PSQL_TYPE_PT = [
    postgresql.INTEGER, postgresql.TIMESTAMP, postgresql.NUMERIC, postgresql.NUMERIC,
    postgresql.NUMERIC, postgresql.NUMERIC, postgresql.NUMERIC, postgresql.NUMERIC,
    postgresql.NUMERIC, postgresql.NUMERIC, postgresql.NUMERIC, postgresql.INTEGER,
    postgresql.INTEGER, postgresql.INTEGER, postgresql.INTEGER, postgresql.INTEGER
]
