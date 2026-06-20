#%%
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '..')) 
import h5py
from h3 import h3
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
import numpy as np
import outlier
import joblib
import importlib
importlib.reload(outlier)
import copy
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from gensim.models import word2vec
import datetime
# %%
class Pipeline():
    def __init__(self, source, file_path, sink, size_trajectory):
        self.source = source
        self.file_path = file_path
        self.sink = sink
        self.scaler_filename = '../../dublin_scaler.save' if self.source == 'dublin' else '../../recife_scaler.save'
        self.s = joblib.load(self.scaler_filename)
        self.size_trajectory = size_trajectory
        self.model_vec = word2vec.Word2Vec(size=30, window=5)

        self.X, self.y = self.load_data(self.file_path)

        self.start()
    
    def start(self):
        print('Starting pipeline...', flush=True)
        # X_tokens, key_set has all the tokens
        X_tokens, key_set = self.get_uber_tokens(self.X)

        print('Splitting data...', flush=True)
        # Split data
        X_train, y_train, X_val, y_val, X_test, y_test = self.get_splitted_data()

        print('Getting uber tokens...', flush=True)
        # Getting tokens for each splitted sample
        X_train_tokens, _ = self.get_uber_tokens(X_train)
        X_val_tokens, _ = self.get_uber_tokens(X_val)
        X_test_tokens, _ = self.get_uber_tokens(X_test)

        print('Generating anomaly ...', flush=True)
        # Generating artificial anomaly for test
        traj_anom_05_01, traj_anom_norm_05_01 = self.generate_outlier(X_test, y_test, 0.5, 0.1)
        traj_anom_1_01, traj_anom_norm_1_01 = self.generate_outlier(X_test, y_test, 1, 0.1)
        traj_anom_05_02, traj_anom_norm_05_02 = self.generate_outlier(X_test, y_test, 0.5, 0.2)
        traj_anom_1_02, traj_anom_norm_1_02 = self.generate_outlier(X_test, y_test, 1, 0.2)
        traj_anom_05_03, traj_anom_norm_05_03 = self.generate_outlier(X_test, y_test, 0.5, 0.3)
        traj_anom_1_03, traj_anom_norm_1_03 = self.generate_outlier(X_test, y_test, 1, 0.3)

        print('Getting uber tokens from anomaly ...', flush=True)
        # Getting tokens for each anomaly sample above
        traj_05_01, key_set_05_01 = self.get_uber_tokens(traj_anom_05_01)
        traj_1_01, key_set_1_01 = self.get_uber_tokens(traj_anom_1_01)
        traj_05_02, key_set_05_02 = self.get_uber_tokens(traj_anom_05_02)
        traj_1_02, key_set_1_02 = self.get_uber_tokens(traj_anom_1_02)
        traj_05_03, key_set_05_03 = self.get_uber_tokens(traj_anom_05_03)
        traj_1_03, key_set_1_03 = self.get_uber_tokens(traj_anom_1_03)

        print('Merging vocabulary ...', flush=True)
        # Merging the tokens vocabulary
        key_set_final = self.get_all_tokens(key_set, key_set_05_01)
        key_set_final = self.get_all_tokens(key_set_final, key_set_1_01)
        key_set_final = self.get_all_tokens(key_set_final, key_set_05_02)
        key_set_final = self.get_all_tokens(key_set_final, key_set_1_02)
        key_set_final = self.get_all_tokens(key_set_final, key_set_05_03)
        key_set_final = self.get_all_tokens(key_set_final, key_set_1_03)

        print('Generating integer tokens ...', flush=True)
        # Generating the integer tokens
        dict_h3_to_int  = {x:idx+1 for idx, x in enumerate(key_set_final)}

        # Getting integer tokens
        X_train_tokens = self.uber_to_label(X_train_tokens, dict_h3_to_int)
        X_train_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_train_tokens, maxlen=self.size_trajectory, padding='post')
        X_val_tokens = self.uber_to_label(X_val_tokens, dict_h3_to_int)
        X_val_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_val_tokens, maxlen=self.size_trajectory, padding='post')
        X_test_tokens = self.uber_to_label(X_test_tokens, dict_h3_to_int)
        X_test_tokens = tf.keras.preprocessing.sequence.pad_sequences(X_test_tokens, maxlen=self.size_trajectory, padding='post')

        traj_05_01 = self.uber_to_label(traj_05_01, dict_h3_to_int)
        traj_05_01 = tf.keras.preprocessing.sequence.pad_sequences(traj_05_01, maxlen=self.size_trajectory, padding='post')
        traj_1_01 = self.uber_to_label(traj_1_01, dict_h3_to_int)
        traj_1_01 = tf.keras.preprocessing.sequence.pad_sequences(traj_1_01, maxlen=self.size_trajectory, padding='post')
        traj_05_02 = self.uber_to_label(traj_05_02, dict_h3_to_int)
        traj_05_02 = tf.keras.preprocessing.sequence.pad_sequences(traj_05_02, maxlen=self.size_trajectory, padding='post')
        traj_1_02 = self.uber_to_label(traj_1_02, dict_h3_to_int)
        traj_1_02 = tf.keras.preprocessing.sequence.pad_sequences(traj_1_02, maxlen=self.size_trajectory, padding='post')
        traj_05_03 = self.uber_to_label(traj_05_03, dict_h3_to_int)
        traj_05_03 = tf.keras.preprocessing.sequence.pad_sequences(traj_05_03, maxlen=self.size_trajectory, padding='post')
        traj_1_03 = self.uber_to_label(traj_1_03, dict_h3_to_int)
        traj_1_03 = tf.keras.preprocessing.sequence.pad_sequences(traj_1_03, maxlen=self.size_trajectory, padding='post')

        print('Training Embedding model ...', flush=True)
        # Training Embedding model
        all_data = np.concatenate(
            (X_train_tokens, traj_05_01
        ))

        self.training_model(all_data)
        del all_data

        print('Getting the embedding for anomaly token trajectories ...', flush=True)
        # Getting the embedding for each sample tokens above
        embed_train = self.getting_embeddings(X_train_tokens)
        embed_val = self.getting_embeddings(X_val_tokens)
        embed_test = self.getting_embeddings(X_test_tokens)

        embed_05_01 = self.getting_embeddings(traj_05_01)
        embed_1_01 = self.getting_embeddings(traj_1_01)
        embed_05_02 = self.getting_embeddings(traj_05_02)
        embed_1_02 = self.getting_embeddings(traj_1_02)
        embed_05_03 = self.getting_embeddings(traj_05_03)
        embed_1_03 = self.getting_embeddings(traj_1_03)

        print('Merging and Saving final data ...', flush=True)
        # Merging lat, lng, lat_norm, lng_norm, timestamp, tokens, embeddings
        X_train_final = np.concatenate(
            (
                X_train, X_train_tokens.reshape((-1,self.size_trajectory,1)), embed_train
            ), axis=-1
        )
        np.save(self.sink+f'/X_train.npy',X_train_final)
        np.save(self.sink+f'/y_train.npy',y_train)
        del X_train_final

        X_val_final = np.concatenate(
            (
                X_val, X_val_tokens.reshape((-1,self.size_trajectory,1)), embed_val
            ), axis=-1
        )
        np.save(self.sink+f'/X_val.npy',X_val_final)
        np.save(self.sink+f'/y_val.npy',y_val)
        del X_val_final

        X_test_final = np.concatenate(
            (
                X_test, X_test_tokens.reshape((-1,self.size_trajectory,1)), embed_test
            ), axis=-1
        )
        np.save(self.sink+f'/X_test.npy',X_test_final)
        np.save(self.sink+f'/y_test.npy',y_test)
        X_test_ord, y_test_ord = self.reord_test_data(X_test_final,y_test)
        np.save(self.sink+f'/X_test_rearranged.npy',X_test_ord)
        np.save(self.sink+f'/y_test_rearranged.npy',y_test_ord)
        del X_test_final

        tokens_05_01 = np.concatenate(
            (
                traj_anom_norm_05_01, traj_05_01.reshape((-1,self.size_trajectory,1)), embed_05_01
            ), axis=-1
        )
        np.save(self.sink+f'/anom_05_01.npy',tokens_05_01)
        del tokens_05_01

        tokens_1_01 = np.concatenate(
            (
                traj_anom_norm_1_01, traj_1_01.reshape((-1,self.size_trajectory,1)), embed_1_01
            ), axis=-1
            
        )
        np.save(self.sink+f'/anom_1_01.npy',tokens_1_01)
        del tokens_1_01

        tokens_05_02 = np.concatenate(
            (
                traj_anom_norm_05_02, traj_05_02.reshape((-1,self.size_trajectory,1)), embed_05_02
            ), axis=-1
        )
        np.save(self.sink+f'/anom_05_02.npy',tokens_05_02)
        del tokens_05_02

        tokens_1_02 = np.concatenate(
            (
                traj_anom_norm_1_02, traj_1_02.reshape((-1,self.size_trajectory,1)), embed_1_02
            ), axis=-1
        )
        np.save(self.sink+f'/anom_1_02.npy',tokens_1_02)
        del tokens_1_02

        tokens_05_03 = np.concatenate(
            (
                traj_anom_norm_05_03, traj_05_03.reshape((-1,self.size_trajectory,1)), embed_05_03
            ), axis=-1
        )
        np.save(self.sink+f'/anom_05_03.npy',tokens_05_03)
        del tokens_05_03

        tokens_1_03 = np.concatenate(
            (
                traj_anom_norm_1_03, traj_1_03.reshape((-1,self.size_trajectory,1)), embed_1_03
            ), axis=-1
        )
        np.save(self.sink+f'/anom_1_03.npy',tokens_1_03)
        del tokens_1_03
    
    # Aqui utilizo para reodendar X_test e y_test de acordo com a ordem das anomalias
    # Quando aplico anomalia, vou pegando em ordem crescente de labels y, ex: 0, 1, 2...
    # Logo preciso ordenar o X_test e o y_test
    def reord_test_data(self, data_x, data_y):
        result_x = None
        result_y = None
        for i in range(self.y.shape[1]):
            idx = np.argwhere(np.argmax(data_y, axis=1)==i).reshape(-1)
            if i == 0:
                result_x = copy.copy(data_x[idx,:,:])
                result_y = copy.copy(data_y[idx,:])
            else:
                aux_x = copy.copy(data_x[idx,:,:])
                aux_y = copy.copy(data_y[idx,:])
                result_x = np.concatenate((result_x,aux_x))
                result_y = np.concatenate((result_y,aux_y))
        return result_x, result_y

    def getting_embeddings(self, data):
        complete_list = list()
        for items in tqdm(data):
            items_list = list()
            for item in items:
                word = str(item)
                if word in self.model_vec:
                    items_list.append(self.model_vec[word].tolist())
                else:
                    items_list.append(np.zeros((30)).tolist())                
            complete_list.append(items_list)
        return np.array(complete_list)

    def training_model(self, data):
        sentences = list()
        for items in tqdm(data):
            sentences.append(list(map(lambda x: str(x), items)))
        
        print('Training model w2v...', flush=True)
        self.model_vec.build_vocab(sentences=sentences)
        self.model_vec.train(sentences=sentences, epochs=20, total_examples=len(sentences))
        print('Saving model w2v...', flush=True)
        self.model_vec.save(self.sink+f'/{self.source}_w2v.model')

    def uber_to_label(self, traj_tokens, dict_h3_to_int):
        final_trajectories = list()
        for i, items in enumerate(traj_tokens):
            point = list()
            for j, item in enumerate(items):
                point.append(dict_h3_to_int[item])
            final_trajectories.append(point)
        return final_trajectories

    def get_all_tokens(self, old_list, new_list):
        set_old = set(old_list)
        set_new = set(new_list)
        diff = set_new.difference(set_old)
        old_list+=diff
        return old_list
    
    def generate_outlier(self,X_test, y_test, dist, percent):
        normalized, not_normalized = None, None
        for i in range(self.y.shape[1]):
            out = outlier.Outlier(i, X_test, y_test, self.s, self.size_trajectory)
            traj_anom_norm, traj_anom = out.get_noise_trajectory(dist, percent)    
            if i == 0:
                not_normalized = traj_anom
                normalized = traj_anom_norm 
            else:
                not_normalized = np.concatenate([not_normalized,traj_anom])
                normalized = np.concatenate([normalized, traj_anom_norm])
        return not_normalized, normalized

    def get_uber_tokens(self,dt):
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

    def get_splitted_data(self):
        X_train,X_test,y_train,y_test = train_test_split(self.X,self.y, test_size=0.2, random_state=1023)
        X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1023)
        return X_train, y_train, X_val, y_val, X_test, y_test


    def load_data(self, file_path):
        data = None       
        
        if self.source == 'dublin':
            with h5py.File(file_path, 'r') as hf:
                    data = hf["data"][:]
                    hf.flush()
                    hf.close()
            X = data[:,:,[1,2,3,4,7]]
            y = data[:,:,0]
            y = [x[0] for x in y]
            y = np.array(y).astype(int)
            y = to_categorical(y-1)
        else:
            X = np.load(file_path, allow_pickle=True)
            y = np.load("../../new_train_data/recife_100_y.npy", allow_pickle=True)                 
            y = to_categorical(y)
        return X, y


#%%
begin_time = datetime.datetime.now()

# source = 'dublin' 
# file_path = "../../new_train_data/dublin_with_embedding_padding_208.hdf5"
# sink = "../dublin_data_from_pipeline/"
# pipe = Pipeline(source, file_path, sink, 208)

source = 'recife' 
file_path = "../../new_train_data/recife_100.npy"
sink = "../recife_data_from_pipeline/"
pipe = Pipeline(source, file_path, sink, 100)

print(datetime.datetime.now() - begin_time) 
# %%
