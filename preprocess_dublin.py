# %%
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import h5py
from h3 import h3
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
import numpy as np
import outlier
import joblib
import importlib
importlib.reload(outlier)
import copy
import matplotlib.pyplot as plt
# %%
scaler_filename = '../../dublin_scaler.save'
s = joblib.load(scaler_filename)
s.mean_
# %%

data = None
with h5py.File("../../new_train_data/dublin_with_embedding_padding_208.hdf5", 'r') as hf:
        data = hf["data"][:]
        hf.flush()
        hf.close()
X = data[:,:,1:5]
y = data[:,:,0]

y = [x[0] for x in y]
y = np.array(y).astype(int)

y = to_categorical(y-1)

#%%
from sklearn.model_selection import train_test_split
def get_splitted_data(X,y):
    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1023)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1023)
    return X_train,y_train, X_val,y_val, X_test, y_test

# %%
def get_uber_tokens(dt):
    traj_tokens = list()
    key_set = set()
    for i, items in tqdm(enumerate(dt)):
        traj = list()
        for j, item in enumerate(items):
            if item[0] != 0. and item[1] != 0.:
                h3_address =  h3.geo_to_h3(item[0],item[1], 10)
                traj.append(h3_address)
                key_set.add(h3_address)
        traj_tokens.append(traj)
    return traj_tokens, list(key_set)
# %%
def uber_to_label(traj_tokens, dict_h3_to_int):
    final_trajectories = list()
    for i, items in enumerate(traj_tokens):
        point = list()
        for j, item in enumerate(items):
            point.append(dict_h3_to_int[item])
        final_trajectories.append(point)
    return final_trajectories
# %%
def save(final_trajectories, name_file):
    fout = open(name_file, 'w')
    for i, traj in enumerate(final_trajectories):
        fout.write("[")
        for point in traj[:-1]:
            fout.write("%s, " % str(point))
        fout.write("%s]\n" % str(traj[-1]))
        
#%%
X_tokens, key_set = get_uber_tokens(X)

X_train, y_train, X_val,y_val, X_test,y_test = get_splitted_data(X,y)

X_tokens_train, _ = get_uber_tokens(X_train)

X_tokens_val, _ = get_uber_tokens(X_val)

X_tokens_test, _ = get_uber_tokens(X_test)

#%% 
# Aqui gero anomalias dos dados de teste e vou colocando em ordem por label ex:0, 1...
for i in range(y.shape[1]):
    out = outlier.Outlier(i, X_test, y_test, s)
    traj_anom_norm, traj_anom = out.get_noise_trajectory(1., 0.3)    
    if i == 0:
        traj_anom_test = traj_anom
    else:
        traj_anom_test = np.concatenate([traj_anom_test,traj_anom])

traj_anom_test[0:1,:,1] == X_test[148:149,:,1]

#%%
def get_all_tokens(old_list, new_list):
    set_old = set(old_list)
    set_new = set(new_list)
    diff = set_new.difference(set_old)
    old_list+=diff
    return old_list
#%%
# Aqui vou chamar get uber tokens para gerar os tokens
X_tokens_test_anom, key_set_test_anom = get_uber_tokens(traj_anom_test)

key_set = get_all_tokens(copy.copy(key_set),copy.copy(key_set_test_anom))

dict_h3_to_int  = {x:idx+1 for idx, x in enumerate(key_set)}

X_traj_test_final_anom = uber_to_label(X_tokens_test_anom, dict_h3_to_int)

X_traj_train_final = uber_to_label(X_tokens_train, dict_h3_to_int)

X_traj_test_final = uber_to_label(X_tokens_test, dict_h3_to_int)

X_traj_val_final = uber_to_label(X_tokens_val, dict_h3_to_int)

#%%
# Aqui salvo todos os dados de treinamento, e validacao(test+anomalia)
save(X_traj_train_final, 'processed_dublin_train.csv')
save(X_traj_val_final, 'processed_dublin_val.csv')
save(X_traj_test_final, 'processed_dublin_test.csv')
save(X_traj_test_final_anom, 'processed_dublin_test_anom.csv')
concat = X_traj_test_final+X_traj_test_final_anom # os índices de anomalias estão na segunda metade
save(concat, 'processed_dublin_test_and_anom.csv')


# %%
# Aqui eu só salvo alguns dados de test e anomalia para experimento controlado
label = 8
count = 0
upper_limit=0
low_limit=0
for i in range(label):
    a = np.argwhere(np.argmax(y_test, axis=1)==i)
    count += np.argwhere(np.argmax(y_test, axis=1)==i).shape[0]
    if i == label-1:
        low_limit = count
        upper_limit = (low_limit + np.argwhere(np.argmax(y_test, axis=1)==i+1).shape[0])

print(low_limit, upper_limit)

idx = np.argwhere(np.argmax(y_test, axis=1)==label).reshape(-1)
X_test_sample = copy.copy(np.array(X_traj_test_final)[idx]).tolist()
X_anom_sample = copy.copy(np.array(X_traj_test_final_anom)[low_limit:upper_limit]).tolist()

concat = X_test_sample[:10]+X_anom_sample[:10] # os índices de anomalias estão na segunda metade
save(concat, 'processed_dublin_test_and_anom_r8.csv')

# %%
trajectories = list()
for i, eachlines in enumerate(open('processed_dublin_test_anom.csv', 'r').readlines()):
    # print(i)
    trajectories.append(eval(eachlines))
# %%
X_traj_test_final = copy.copy(trajectories)
#%%
X_traj_test_final_anom = copy.copy(trajectories)

#%%
_, y_train, _,y_val, _,y_test = get_splitted_data(X,y)
# %%
