import pandas as pd

treshold = 50
coordinates = {'dublin': ['lng', 'lat']}
dataset = 'dublin'

def clean_data(data):
    data = adjust_types(data)
    return data

def adjust_types(data):
    data.stop = data.stop.astype('bool')
    data.congestion = data.congestion.astype('bool')
    data['datetime'] = pd.to_datetime(data["timestamp"], unit='us')
    data['datetime_hour'] = data["datetime"].dt.hour
    data['datetime_day'] = data["datetime"].dt.day
    return data

def unique_trajectories_tuple(data, features):
    trajectories = data.groupby(features).size().reset_index()
    trajectories = trajectories[trajectories[0] > treshold]
    return trajectories[features].values


def _has_min_quantity_of_points(items):
    return len(items) > 10

def is_window(delta_time):
    return delta_time < 5 * 60

def delta_time(t1, t2) -> float:
    ##Return time difference between time in seconds
    t1 = pd.to_datetime(t1,unit='us')
    t2 = pd.to_datetime(t2,unit='us')
    delta = pd.Timedelta(np.abs(t2 - t1))
    return delta.seconds

def create_sentences(data, trajectory_features) -> list:
#    data = data.sort_values('timestamp')
    actual_trajectory = data[trajectory_features].iloc[0].values
    actual_time = data.iloc[0].timestamp
    partial_list, complete_trajectory = [], []
    iterator = 0
    for index in tqdm(data.index):
        if (is_valid_point(data, index, actual_trajectory)):
            delta = delta_time(actual_time, data.loc[index].timestamp)
            if is_window(delta):
                partial_list.append(data.loc[index].tolist())
        actual_time = data.loc[index].timestamp
        iterator += 1
    if self._has_min_quantity_of_points(partial_list):
        complete_trajectory.append(partial_list)
    return complete_trajectory

def is_valid_point(data, idx, actual_trajectory):
    return all(data[trajectory_features].loc[idx] == actual_trajectory)
