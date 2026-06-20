#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm import tqdm
import tensorflow as tf
import joblib
import h5py
from copy import copy
# %%
raw_traj = np.load('../../new_train_data/recife_ts.npy', allow_pickle=True)
# %%
for item in raw_traj:
    item[:,4] = np.apply_along_axis(lambda x: x[4].timestamp(), 1, item)

result = list()
for item in tqdm(raw_traj):
    # item[:,:] = np.apply_along_axis(lambda x: x.astype(np.float32), 0, item)
    result.append(copy(item[:,1:].astype(np.float64)))

raw_traj = copy(result)
# %%
## Using scler into lat, lng and timestamp
std = StandardScaler()
for item in tqdm(raw_traj):
    std.partial_fit(item[:,1:])
# %%
## Here, I apply the scaler
new_traj = list()
for item in tqdm(raw_traj):
    aux = std.transform(item[:,1:])
    new_traj.append(np.concatenate([item,aux],axis=1))
# %%
## Here, I remove desncessary features (id, timestamp not normalized)
final_traj = list()
for item in tqdm(new_traj):
    final_traj.append(item[:,[0,1,2,4,5,6]])

#%%
## Here, I select label's trajectory with more than 400 exemamples
dict_label = {lb[0,0]:0 for lb in final_traj}
for trj in final_traj:
    dict_label[trj[0,0]]+=1

traj_big_sample = list()
for trj in final_traj:
    if dict_label[trj[0,0]] >= 400:
        traj_big_sample.append(trj)

# %%
items_len = list(map(lambda x: len(x),traj_big_sample))
print(f'Mean:{np.mean(items_len)}\tStd:{np.std(items_len)}\tMin:{np.min(items_len)}\tMax:{np.max(items_len)}')

# %%
## Here, I put a padding
final_traj = tf.keras.preprocessing.sequence.pad_sequences(traj_big_sample, dtype='float64', maxlen=128, padding='post', value=0.0)
# final_traj = list()
# for items in tqdm(traj_big_sample):
#     if items.shape[0] > 150:
#         aux = copy(items[:150,:].tolist())
#         final_traj.append(items[:150,:])
#     elif items.shape[0] < 150:
#         diff = 150 - items.shape[0]
#         label = items[0,0]
#         pad = [[label, 0.0, 0.0, 0.0, 0.0, 0.0]]*diff
#         aux = items.tolist().append(copy(pad))
#         final_traj.append(aux)
#     else:
#         aux = copy(items.tolist())
#         final_traj.append(aux)

# final_traj = np.array(final_traj)
# %%
## Here, I replace real label by sequence labels
fake_labels = {lb:np.float32(i) for i, lb in enumerate(np.unique(final_traj[:,:,0]))}
for item in final_traj:
    item[:,0] = [fake_labels[item[0,0]]]*item.shape[0]    
# %%
# Saving
# with h5py.File("../../new_train_data/recife_170.hdf5", "w") as f:
#      dset = f.create_dataset("data", data=final_traj)
#      f.flush()
#      f.close()
np.save("../../new_train_data/recife_150.npy",final_traj)

joblib.dump(std, '../../recife_scaler.save')
# %%
