import pandas as pd
import numpy as np
import sys
from tqdm import tqdm

# data = pd.read_csv('../../pre_processamento/generate_labels/data-2000000_bus_stop_traffic_light.csv')
# features=['lat','lng','instante','rota','velocidade','posicao','viaje','matricula_id','lat_uber','lng_uber','label']
data = pd.read_csv('../../dublin_2013_4M.csv')
features=['lat','lng','instante','rota','matricula_id']

def delta_time(t1,t2)->float:
##Return time difference between time in minutes
    t1 = pd.to_datetime(t1)
    t2 = pd.to_datetime(t2)
    delta = pd.Timedelta(np.abs(t2-t1))
    return delta.days*24*60 + delta.seconds/60
    
def get_element_by_element(_id, data):
        row = data[data['id']==_id]
        row = row[features].values[0]
        return row.tolist()
    
def is_window(delta_time):
    return delta_time < 5


def _has_min_quantity_of_points(items):
    return len(items) > 10

def is_same_trajectory(idx, data, old_viaje, old_time, old_matricula, old_rota, dub=False):
    if dub:
        if (data.at[idx,'matricula_id'] == old_matricula) and \
           (is_window(delta_time(old_time,data.at[idx,'instante']))) and \
            (data.at[idx,'rota'] == old_rota):
            return True
    else:
        if (data.at[idx,'matricula_id'] == old_matricula) and \
           (data.at[idx,'viaje'] == old_viaje) and \
           (is_window(delta_time(old_time,data.at[idx,'instante']))) and \
            (data.at[idx,'rota'] == old_rota):
            return True

    return False
        
    


db = False
dublin = True

def create_sentence_of_trajectory(data):
    old_matricula = data.iloc[0].matricula_id
    
#     old_viaje = data.iloc[0].viaje
    
    old_viaje = 'nothing'
    
    old_time = data.iloc[0].instante
    
    old_rota = data.iloc[0].rota
    
    len_sentence = []
    
    partial_list, complete_list = [], []
    
    iterator = 0
    
    for idx in tqdm(data.index):
        if is_same_trajectory(idx, data, old_viaje, old_time,old_matricula, old_rota, dublin):
        
            partial_list.append(get_element_by_element(data.at[idx,'id'], data))
        else:
            if _has_min_quantity_of_points(partial_list):
                len_sentence.append(len(partial_list))
#                     complete_list.append(partial_list)
                if db:
                    self.table.insert_one({'st':partial_list})
                else:
                    complete_list.append(partial_list)

            partial_list = []
            partial_list.append(get_element_by_element(data.at[idx,'id'], data))


        old_matricula = data.at[idx,'matricula_id']
#         old_viaje = data.at[idx,'viaje']
        old_viaje = 'nothing'
        old_time = data.at[idx,'instante']
        old_rota = data.at[idx,'rota']
        iterator +=1


    if _has_min_quantity_of_points(partial_list):
        if db:
            table.insert_one({'st':partial_list})
        else:
            complete_list.append(partial_list)
        len_sentence.append(len(partial_list))

#     if not db:
#         np.save('data_sentence_without_tqdm',np.array(self.pad_sentences(complete_list,max(len_sentence))))
    
    print(iterator)
    return complete_list

final_list = create_sentence_of_trajectory(data)
print('Saving...')
np.save('sentences_dublin_4M.npy',final_list)