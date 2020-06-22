import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import optuna
import sklearn.ensemble
import sklearn.model_selection
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import lightgbm as lgb
import optuna, os, uuid, pickle

# cache_final_all_classes_dub.hdf5

# with h5py.File('../../cache_final_4_classes.hdf5', 'r') as hf:
#     list_stat_x_b = hf['list_stat_x_b'][:]
#     list_stat_x_a = hf['list_stat_x_a'][:]
#     list_stat_x_c = hf['list_stat_x_c'][:]
#     list_stat_x_bs = hf['list_stat_x_bs'][:]
#     list_stat_x_as = hf['list_stat_x_as'][:]
#     list_stat_y = hf['list_stat_y'][:]
    

# from sklearn.preprocessing import StandardScaler
# scaler = StandardScaler()

# def scaler(data,it, dim=3):
#     scaler = StandardScaler()
#     if dim == 3:
#         for i in tqdm([0,1,2,3,4,5,6,9]):
#             data[:,:,i]= scaler.fit_transform(data[:,:,i])
#     else:
#         data= scaler.fit_transform(data)
#     return data

# list_stat_x_b = scaler(list_stat_x_b,list_stat_x_b.shape[2])
# list_stat_x_a = scaler(list_stat_x_a,list_stat_x_a.shape[2])
# # list_stat_x_c = scaler(list_stat_x_c,list_stat_x_c.shape[1],2)
# list_stat_x_bs = scaler(list_stat_x_bs, list_stat_x_bs.shape[1],2)
# list_stat_x_as = scaler(list_stat_x_as, list_stat_x_as.shape[1],2)

# scaler = StandardScaler()
# list_stat_x_c[:,[0,1,2,3,4,5,6,9]] = scaler.fit_transform(list_stat_x_c[:,[0,1,2,3,4,5,6,9]])

# X = list()
# for i in range(len(list_stat_x_b)):
#     X.append([list_stat_x_b[i],list_stat_x_a[i],list_stat_x_c[i],list_stat_x_bs[i],list_stat_x_as[i]])

# y = np.array(list_stat_y) if type(list_stat_y)!=np.ndarray else list_stat_y
# y = y.astype(int)
# ../cache_embeddings_STOD_dublin.hdf5
embeddings = None
y = None
n_class = 66
# data_path = '../lstm/cache_embeddings_to_baselines_dublin.hdf5'
data_path = '../cache_embeddings_STOD_dublin_clean.hdf5'
with h5py.File(data_path, 'r') as hf:
    embeddings = hf['embeddings_predicted'][:]
    y = hf['y'][:]
#     y_label = hf['y_encoder'][:]
    hf.flush()
    hf.close()

from sklearn.decomposition import pca

pc = pca.PCA(32)
embeddings = pc.fit_transform(embeddings)


from sklearn.model_selection import train_test_split
# X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1023)
X_train,X_test,y_train,y_test = train_test_split(embeddings, y,test_size=0.2, random_state=1023)
X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1023)

X_train_final = X_train[:50000]
y_train_final = y_train[:50000]
X_val_final = X_val[:10000]
y_val_final = y_val[:10000]

# Realizar teste com raw data
# def get_one_batch2(x, y)->tuple():
#         list_x_b, list_x_a = list(), list()
#         list_x_target, list_stat_b, list_stat_a = list(), list(), list()

#         x1 = x


#         for j in tqdm(range(len(x))):
#             list_x_b.append(x1[j][0])
#             list_x_a.append(x1[j][1])
#             list_x_target.append(x1[j][2][:])
#             list_stat_b.append(x1[j][3])
#             list_stat_a.append(x1[j][4])
        
#         a_x_b = np.array(list_x_b)
#         a_x_a = np.array(list_x_a)
#         x_t = np.array(list_x_target)
#         stat_b = np.array(list_stat_b)
#         stat_a = np.array(list_stat_a)
        
#         return [a_x_b[:,:,[0,1,2,3,4,5,6,9]],a_x_b[:,:,7],a_x_b[:,:,8],
#                 a_x_a[:,:,[0,1,2,3,4,5,6,9]],a_x_a[:,:,7],a_x_a[:,:,8],
#                 x_t[:,[0,1,2,3,4,5,6,9]], x_t[:,7], x_t[:,8],
#                 stat_b,
#                 stat_a],y

# get_batches_train = get_one_batch2(X_train, y_train)
# get_batches_val = get_one_batch2(X_val, y_val)
# get_batches_test = get_one_batch2(X_test,y_test)

# input_data_b = get_batches_train[0][0].reshape((get_batches_train[0][0].shape[0], 16*8))
# input_data_a = get_batches_train[0][3].reshape((get_batches_train[0][3].shape[0], 16*8))

# input_data_b_test = get_batches_test[0][0].reshape((get_batches_test[0][0].shape[0], 16*8))
# input_data_a_test = get_batches_test[0][3].reshape((get_batches_test[0][3].shape[0], 16*8))

# input_data_b_val = get_batches_val[0][0].reshape((get_batches_val[0][0].shape[0], 16*8))
# input_data_a_val = get_batches_val[0][3].reshape((get_batches_val[0][3].shape[0], 16*8))

# input_data = np.concatenate((input_data_b,get_batches_train[0][1], 
#                              get_batches_train[0][2],input_data_a,
#                              get_batches_train[0][4],get_batches_train[0][5],
#                              get_batches_train[0][6],get_batches_train[0][7].reshape((get_batches_train[0][7].shape[0],1)),
#                              get_batches_train[0][8].reshape((get_batches_train[0][8].shape[0],1)),get_batches_train[0][9],
#                              get_batches_train[0][10]), axis=1)
# input_data_val = np.concatenate((input_data_b_val,get_batches_val[0][1], 
#                              get_batches_val[0][2],input_data_a_val,
#                              get_batches_val[0][4],get_batches_val[0][5],
#                              get_batches_val[0][6],get_batches_val[0][7].reshape((get_batches_val[0][7].shape[0],1)),
#                              get_batches_val[0][8].reshape((get_batches_val[0][8].shape[0],1)),get_batches_val[0][9],
#                              get_batches_val[0][10]), axis=1)

# input_data_test = np.concatenate((input_data_b_test,get_batches_test[0][1], 
#                              get_batches_test[0][2],input_data_a_test,
#                              get_batches_test[0][4],get_batches_test[0][5],
#                              get_batches_test[0][6],get_batches_test[0][7].reshape((get_batches_test[0][7].shape[0],1)),
#                              get_batches_test[0][8].reshape((get_batches_test[0][8].shape[0],1)),get_batches_test[0][9],
#                              get_batches_test[0][10]), axis=1)

# y_train = get_batches_train[1]
# y_val = get_batches_val[1]
# y_test = get_batches_test[1]

# from sklearn.decomposition import pca

# pc = pca.PCA(32)
# X_train_final = pc.fit_transform(input_data[:50000])
# y_train_final = y_train[:50000]

# pc = pca.PCA(32)
# X_val_final = pc.fit_transform(input_data_val[:10000])
# y_val_final = y_val[:10000]

# pc = pca.PCA(32)
# X_test_final = pc.fit_transform(input_data_test)


def solver(X,y, model):
    return sklearn.model_selection.cross_val_score(
        model, X_train_final, y_train_final, n_jobs=-1, cv=3).mean()

def objective_randomf(trial):
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr("uuid", trial_uuid)
    
    n_estimators = trial.suggest_int('n_estimators', 2, 50)
    max_depth = int(trial.suggest_loguniform('max_depth', 1, 32))
        
    clf = sklearn.ensemble.RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth)
    
    if not os.path.exists("random_output"):
        os.mkdir("random_output")
    with open("random_output/"+f"{trial_uuid}.pkl", "wb") as fp:
        pickle.dump(clf, fp)
    
    return solver(X_train_final, y_train_final,clf)

def objective_svm(trial):
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr("uuid", trial_uuid)
    
    svm_c = trial.suggest_loguniform('C', 1e-10, 1e10)
        
    clf = LinearSVC(C=svm_c,verbose=1) 
    
    if not os.path.exists("svm_output"):
        os.mkdir("svm_output")
    with open("svm_output/"+f"{trial_uuid}.pkl", "wb") as fp:
        pickle.dump(clf, fp)
 
    
    return solver(X_train_final, y_train_final,clf)

def objective_gaussianb(trial):
    clf = GaussianNB()
    return solver(X_train_final, y_train_final,clf)

def objective_knn(trial):
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr("uuid", trial_uuid)

    neighbors = trial.suggest_int('n_neighbors',1,30)
    algorithm = trial.suggest_categorical('algorithm',['auto', 'ball_tree', 'kd_tree', 'brute'])
    clf = KNeighborsClassifier(n_neighbors=neighbors, algorithm=algorithm)

    if not os.path.exists("knn_output"):
        os.mkdir("knn_output")
    with open("knn_output/"+f"{trial_uuid}.pkl", "wb") as fp:
        pickle.dump(clf, fp)
        
    return solver(X_train_final, y_train_final,clf)

def objective_lgbm(trial):
    trial_uuid = str(uuid.uuid4())
    trial.set_user_attr("uuid", trial_uuid)
    
    dtrain = lgb.Dataset(X_train_final, label=y_train_final)
    dvalidation = lgb.Dataset(X_val_final, label=y_val_final)

    params = {
            'n_estimators':trial.suggest_int('n_estimators', 2, 50),
            'boosting_type': trial.suggest_categorical('boosting', ['gbdt', 'dart', 'goss']),
            'objective': 'multiclass',
            'num_class':n_class,
            'max_bin':63,
            'num_threads':7,
            'pred_early_stop':True,
            'force_row_wise':True,
            'save_binary':True,
            'is_unbalance':True,
            'metric': 'multi_logloss',
            'num_leaves': trial.suggest_int("num_leaves", 10, 150),
            'learning_rate': trial.suggest_loguniform("learning_rate", 1e-5, 1),
            'feature_fraction': trial.suggest_uniform("feature_fraction", 0.0, 1.0),
            'verbose' : 0
        }
    
    if params['boosting_type'] == 'dart':
        params['drop_rate'] = trial.suggest_loguniform('drop_rate', 1e-8, 1.0)
        params['skip_drop'] = trial.suggest_loguniform('skip_drop', 1e-8, 1.0)
    if params['boosting_type'] == 'goss':
        params['top_rate'] = trial.suggest_uniform('top_rate', 0.0, 1.0)
        params['other_rate'] = trial.suggest_uniform('other_rate', 0.0, 1.0 - params['top_rate'])
        
    
    pruning_callback = optuna.integration.LightGBMPruningCallback(trial, "multi_logloss")
    
    
    gbm = lgb.train(
        params, dtrain, valid_sets=[dvalidation], verbose_eval=False)
    
    y_pred_train = [np.argmax(line) for line in gbm.predict(X_train_final)]
    y_pred_validation = [np.argmax(line) for line in gbm.predict(X_val_final)]
    
    error_train =1 - accuracy_score(y_train_final, y_pred_train)
    error_val = 1 - accuracy_score(y_val_final, y_pred_validation)
    
    if not os.path.exists("lgb_output"):
        os.mkdir("lgb_output")
    with open("lgb_output/"+f"{trial_uuid}.pkl", "wb") as fp:
        pickle.dump(gbm, fp)
    
    return error_val


study = optuna.create_study(direction='maximize') # Aqui uso para os demais pq quero maximizar a acurácia
# study = optuna.create_study() # Aqui uso para lgbm pq quero minimizar
study.optimize(objective_randomf, n_trials=120)

trial = study.best_trial

print('Accuracy: {}'.format(trial.value))
print("Best hyperparameters: {}".format(trial.params))

print(study.best_trial.user_attrs)
df = study.trials_dataframe()
# df.to_csv("optuna_lgb_dublin_stod.csv")
# df.to_csv("optuna_knn_dublin_stod.csv")
df.to_csv("optuna_random_dublin_stod.csv")
# df.to_csv("optuna_svm_dublin_stod.csv")