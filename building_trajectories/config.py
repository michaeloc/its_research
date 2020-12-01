from sqlalchemy import create_engine
from secrets import pqsq_passwd

engine = create_engine( f'postgresql://postgres:{pqsq_passwd}@localhost/urbanmobility')

query_str = "line_id == {} & journey_id == '{}' &  time_frame == '{}' & vehicle_journey_id == {} & operator == '{}' & vehicle_id == {}"

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
