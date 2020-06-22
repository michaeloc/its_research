import h5py
import numpy as np
import pandas as pd
import autosklearn.classification
from tqdm import tqdm

with h5py.File('../../cache_final_all_classes_dub.hdf5', 'r') as hf:
    list_stat_x_b = hf['list_stat_x_b'][:]
    list_stat_x_a = hf['list_stat_x_a'][:]
    list_stat_x_c = hf['list_stat_x_c'][:]
    list_stat_x_bs = hf['list_stat_x_bs'][:]
    list_stat_x_as = hf['list_stat_x_as'][:]
    list_stat_y = hf['list_stat_y'][:]
    

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def scaler(data,it, dim=3):
    scaler = StandardScaler()
    if dim == 3:
        for i in tqdm([0,1,2,3,4,5,6,9]):
            data[:,:,i]= scaler.fit_transform(data[:,:,i])
    else:
        data= scaler.fit_transform(data)
    return data

list_stat_x_b = scaler(list_stat_x_b,list_stat_x_b.shape[2])
list_stat_x_a = scaler(list_stat_x_a,list_stat_x_a.shape[2])
# list_stat_x_c = scaler(list_stat_x_c,list_stat_x_c.shape[1],2)
list_stat_x_bs = scaler(list_stat_x_bs, list_stat_x_bs.shape[1],2)
list_stat_x_as = scaler(list_stat_x_as, list_stat_x_as.shape[1],2)

scaler = StandardScaler()
list_stat_x_c[:,[0,1,2,3,4,5,6,9]] = scaler.fit_transform(list_stat_x_c[:,[0,1,2,3,4,5,6,9]])

X = list()
for i in range(len(list_stat_x_b)):
    X.append([list_stat_x_b[i],list_stat_x_a[i],list_stat_x_c[i],list_stat_x_bs[i],list_stat_x_as[i]])

y = np.array(list_stat_y) if type(list_stat_y)!=np.ndarray else list_stat_y
y = y.astype(int)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1023)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1023)

def get_one_batch2(x, y)->tuple():
        list_x_b, list_x_a = list(), list()
        list_x_target, list_stat_b, list_stat_a = list(), list(), list()

        x1 = x
#         y1 = K.one_hot(y,4)
#         y1 = K.reshape(y1,(len(y),-1))
#         with tf.Session() as sess:
#             y1 = sess.run(y1)

        for j in tqdm(range(len(x))):
            list_x_b.append(x1[j][0])
            list_x_a.append(x1[j][1])
#             list_x_target.append(x1[j][2][:10])
            list_x_target.append(x1[j][2][:])
            list_stat_b.append(x1[j][3])
            list_stat_a.append(x1[j][4])
        
        a_x_b = np.array(list_x_b)
        a_x_a = np.array(list_x_a)
        x_t = np.array(list_x_target)
        stat_b = np.array(list_stat_b)
        stat_a = np.array(list_stat_a)
        
        return [a_x_b[:,:,[0,1,2,3,4,5,6,9]],a_x_b[:,:,7],a_x_b[:,:,8],
                a_x_a[:,:,[0,1,2,3,4,5,6,9]],a_x_a[:,:,7],a_x_a[:,:,8],
                x_t[:,[0,1,2,3,4,5,6,9]], x_t[:,7], x_t[:,8],
                stat_b,
                stat_a],y
#     ,x_t[:,:]
# [0,1,2,3,4,5,6,9]
# x_t no final para pegar o id de cada ponto e construir a matriz

get_batches_train = get_one_batch2(X_train, y_train)
get_batches_test = get_one_batch2(X_test,y_test)

input_data_b = get_batches_train[0][0].reshape((1087373, 16*8))
input_data_a = get_batches_train[0][3].reshape((1087373, 16*8))

input_data_b_test = get_batches_test[0][0].reshape((get_batches_test[0][0].shape[0], 16*8))
input_data_a_test = get_batches_test[0][3].reshape((get_batches_test[0][3].shape[0], 16*8))

input_data = np.concatenate((input_data_b,get_batches_train[0][1], 
                             get_batches_train[0][2],input_data_a,
                             get_batches_train[0][4],get_batches_train[0][5],
                             get_batches_train[0][6],get_batches_train[0][7].reshape((get_batches_train[0][7].shape[0],1)),
                             get_batches_train[0][8].reshape((get_batches_train[0][8].shape[0],1)),get_batches_train[0][9],
                             get_batches_train[0][10]), axis=1)

input_data_test = np.concatenate((input_data_b_test,get_batches_test[0][1], 
                             get_batches_test[0][2],input_data_a_test,
                             get_batches_test[0][4],get_batches_test[0][5],
                             get_batches_test[0][6],get_batches_test[0][7].reshape((get_batches_test[0][7].shape[0],1)),
                             get_batches_test[0][8].reshape((get_batches_test[0][8].shape[0],1)),get_batches_test[0][9],
                             get_batches_test[0][10]), axis=1)

y_train = get_batches_train[1]
y_test = get_batches_test[1]

from sklearn.decomposition import pca

pc = pca.PCA(32)
X_train_final = pc.fit_transform(input_data[:5000])
y_train_final = y_train[:5000]

pc = pca.PCA(32)
X_test_final = pc.fit_transform(input_data_test)

automl = autosklearn.classification.AutoSklearnClassifier(
    include_estimators=["random_forest","gaussian_nb","libsvm_svc","k_nearest_neighbors", ],
    ensemble_size=50,
    n_jobs=4,
    exclude_estimators=None,
    include_preprocessors=["no_preprocessing", ], exclude_preprocessors=None)

automl.fit(X_train_final, y_train_final)
predict = automl.predict(X_test_final)

from sklearn.metrics import f1_score, recall_score, precision_score
print(f1_score(y_test, predict, average='weighted'))
print(recall_score(y_test, predict, average='weighted'))
print(precision_score(y_test, predict, average='weighted'))

print(automl.cv_results_)

print(automl.show_models())

print(automl.sprint_statistics())