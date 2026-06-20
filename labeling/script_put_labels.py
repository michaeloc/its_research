import pandas as pd
import numpy as np
import datetime
from tqdm import tqdm
import os.path
import preprocess as prep
from multiprocessing import Pool
import time
import copy

#
# Recife
# GTFS_STOP_PATH = '../../GTFSjaneiro/stops.txt'
# GTFS_SHAPE = '../../GTFSjaneiro/shapes.txt'
# GTFS_TRIP = '../../GTFSjaneiro/trips_modified.csv'
# DATA_PATH = '../../sentences_6M.npy'
# DATA_PATH = '../../gtfs_sentences.npy'
# SEMAFORO_PATH='../../GTFSjaneiro/semaforos.csv'

#Dublin

GTFS_STOP_PATH = '../../dublin/bus_stops.csv'
DATA_PATH = '../../sentences_dublin_4M_to_get_label.npy'
SEMAFORO_PATH='../../dublin/traffic_signals.csv'

sentences = np.load(DATA_PATH)
stops = pd.read_csv(GTFS_STOP_PATH)
trfl = pd.read_csv(SEMAFORO_PATH)

trfl.dropna(inplace=True)

array_stops = stops.values
array_trfl = trfl.values

prepro = prep.PreProcess()

def adding_bust(data):
    chunck = []
    for items in tqdm(data):
        final_item = []
        for item in items:
            for stop in array_stops:
                #Recife
#                 dist = prepro.distance_in_meters([item[0],item[1]], [stop[4],stop[5]])
                #Dublin
                dist = prepro.distance_in_meters([item[0],item[1]], [stop[1],stop[2]])
                if item[4] < 5 and dist < 15:
                    print('bustop')
                    item.append('bus_stop')
                    break
            final_item.append(item)
        chunck.append(final_item)
    return chunck

def task1():
    chunks = [sentences[i:i+200] for i in range(0, len(sentences),200)]
    pool = Pool(processes=len(chunks))
    result = pool.map_async(adding_bust,chunks)
    while not result.ready():
        print("Running...")
        time.sleep(0.5)
    pool.terminate()
    return result

print('Iniciando label stop')
a = task1()
b =a.get()
print('Finalizando label stop')

def adding_tfl(data):
    chunck = []
    for items in tqdm(data):
        final_item = []
        for item in items:
            for stop in array_trfl:
                if len(item)==10:
                # Recife
                # dist = prepro.distance_in_meters([item[0],item[1]], [stop[7],stop[8]])
                # Dublin
                    dist = prepro.distance_in_meters([item[0],item[1]], [stop[1],stop[2]])
                    if item[4] < 5 and dist < 30:
                        item.append('traffic_light')
                        break
            final_item.append(item)
        chunck.append(final_item)
    return chunck

def task2(b):
    chunks = b
    pool = Pool(processes=200)
    result = pool.map_async(adding_tfl,chunks)
    while not result.ready():
        print("Running...")
        time.sleep(0.5)
    pool.terminate()
    return result

print('Iniciando label trfl')
c = task2(b)
d = c.get()
print('Finalizando label trfl')

print('putting the label in_route')
for elements in tqdm(d):
    for items in elements:
        for item in items:
            if len(item) == 10:
                item.append('in_route')


print('saving')
# Recife
# np.save('../../gtfs_sentences_labels',d)
# Dublin
np.save('dublin_sentences_labels',d)


