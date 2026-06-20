import numpy as np
import tensorflow as tf
from keras.layers import LSTM, Input, Dense, Reshape, Conv1D, MaxPool1D, Dropout, Flatten, BatchNormalization, Embedding,GRU
from keras.models import Model, Sequential
from keras.layers.merge import concatenate,average
from keras.backend import mean
from keras import regularizers
from keras import optimizers
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint
import sys
from tqdm import tqdm
sys.path.insert(1, '/home/mobility/michael/segmentation/mobility_its')
from loss import Losses


import datetime

class Stod():
    def __init__(self, space_params):
        self.params = space_params
        self.seq = self.params['seq']
        self.features = self.params['features']
        self.loss = Losses()
    
    def build_model(self):
        sequence_input = Input(shape=(self.seq, self.features), dtype='float32', name='sequence_input')
        gru = GRU(self.params['gru'], activation='relu', name='GRU')(sequence_input)

        fc1 = Dense(self.params['d1'], activation='relu')(gru)
        fc1 = Dropout(self.params['drp'])(fc1)
        fc2 = Dense(self.params['classes'], activation='softmax')(fc1)

        model = Model(sequence_input, fc2)

        adam = optimizers.Adam(lr=self.params['lr'])
        model.compile(loss=self.loss.focal_loss(),
                      optimizer=self.params['op'],
                      metrics=['acc'])

        return model

    def fit_model(self, model, inputs):
        
        checkpoint = ModelCheckpoint(self.params['model_name'], 
                                 monitor='val_acc',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='max')
        
        callbacks_list = [checkpoint]
        
        result = model.fit(x=inputs['x_train'],
                           y=inputs['y_train'], 
                           batch_size=self.params['batch'], 
                           epochs=self.params['epochs'],
                           verbose=1, 
                           validation_data=[inputs['x_val'],inputs['y_val']],
                           callbacks=callbacks_list)
            
        return result, model

    def inverse_transform(self, data):
        classes = list()
        for item in tqdm(data):
            label_idx = np.argmax(item, axis=0).tolist()
            classes.append(label_idx)
        return classes

    def print_scores(self, model, inputs):
        predicted = model.predict(inputs['x_test'], verbose=1)
        y_pred = self.inverse_transform(predicted)
        y_true = self.inverse_transform(inputs['y_test'])
        wf1 = f1_score(y_true, y_pred, average="weighted")
        wr = recall_score(y_true, y_pred, average="weighted")
        wp = precision_score(y_true, y_pred, average="weighted")
        print(f'WF1- {wf1}')  
        print(f'WR- {wr}')  
        print(f'WP- {wp}')
        return {'wf1':wf1, 'wr':wr, 'wp': wp}



