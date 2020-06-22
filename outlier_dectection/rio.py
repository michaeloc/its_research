import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Input, Dense, Reshape, Conv1D, MaxPool1D, Dropout, Flatten, BatchNormalization, Embedding,GRU
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.backend import mean
from tensorflow.keras import regularizers
from tensorflow.keras import optimizers
from sklearn.metrics import f1_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
import sys
from tqdm import tqdm
sys.path.insert(1, '/home/mobility/michael/segmentation/mobility_its')
from loss import Losses

class Rio():
    def __init__(self, space_params):
        self.params = space_params
        self.seq = 100
        self.features = 3
        self.loss = Losses()
    
    def build_model(self):
        
        sequence_input = Input(shape=(self.seq, self.features), dtype='float32', name='sequence_input')

        conv_a = Conv1D(self.params['conv1'],kernel_size=3, activation='relu', padding='same')(sequence_input)
        conv_a = Conv1D(self.params['conv2'],kernel_size=3, activation='relu', padding='same')(conv_a)
        conv_a = Conv1D(self.params['conv3'],kernel_size=3, activation='relu', padding='same')(conv_a)
        maxp = MaxPool1D(2,2)(conv_a)
        conv_a = Conv1D(self.params['conv4'],kernel_size=3, activation='relu', padding='same')(maxp)
        conv_a = Conv1D(self.params['conv5'],kernel_size=3, activation='relu', padding='same')(conv_a)
        conv_a = Conv1D(self.params['conv6'],kernel_size=3, activation='relu', padding='same')(conv_a)
        maxp = MaxPool1D(2,2)(conv_a)

        flt = Flatten()(maxp)

        fc1 = Dense(self.params['d1'], activation='relu')(flt)
        fc2 = Dense(self.params['d2'], activation='relu')(fc1)

        fc3 = Dense(self.params['classes'], activation='softmax')(fc2)

        model = Model(sequence_input, fc3)

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




