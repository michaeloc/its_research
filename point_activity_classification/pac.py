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
from attention import Attention
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
import sys
from tqdm import tqdm
sys.path.insert(1, '/home/mobility/michael/segmentation/its_research/')
from utility.loss import Losses


import datetime

class Pac():
    def __init__(self, space_params):
        self.params = space_params
        self.seq = self.params['seq']
        self.features = self.params['features']
        self.loss = Losses()
    
    def build_model(self):
        self.sequence_input_b = Input(shape=(self.seq, self.features), dtype='float32', name='sequence_input_a')
        self.sequence_input_a = Input(shape=(self.seq, self.features), dtype='float32', name='sequence_input_b')
        self.x_to_be_predicted = Input(shape=(self.features,), dtype='float32', name='x_to_be_predicted')
        self.sequence_input_stat_a = Input(shape=(self.params['features_stat'],), dtype='float32', name='sequence_input_stat_a')
        self.sequence_input_stat_b = Input(shape=(self.params['features_stat'],), dtype='float32', name='sequence_input_stat_b')

        embedding_layer_w = Embedding(7,30)
        embedding_layer_h = Embedding(24,30)

        self.inputs_w_b = Input(shape=(self.seq,), dtype='int32', name='input_w')
        self.inputs_h_b = Input(shape=(self.seq,), dtype='int32', name='input_h')

        self.inputs_w_a = Input(shape=(self.seq,), dtype='int32', name='input_w_a')
        self.inputs_h_a = Input(shape=(self.seq,), dtype='int32', name='input_h_a')

        self.inputs_w_x = Input(shape=(1,), dtype='int32', name='input_w_x')
        self.inputs_h_x = Input(shape=(1,), dtype='int32', name='input_h_x')

        embed_w_b = embedding_layer_w(self.inputs_w_b)
        embed_w_r_b = Reshape((self.seq,30))(embed_w_b)
        embed_w_a = embedding_layer_w(self.inputs_w_a)
        embed_w_r_a = Reshape((self.seq,30))(embed_w_a)

        embed_h_b = embedding_layer_h(self.inputs_h_b)
        embed_h_r_b = Reshape((self.seq,30))(embed_h_b)
        embed_h_a = embedding_layer_h(self.inputs_h_a)
        embed_h_r_a = Reshape((self.seq,30))(embed_h_a)

        embed_w_x = embedding_layer_w(self.inputs_w_x)
        embed_h_x = embedding_layer_h(self.inputs_h_x)


        concat_b = concatenate([self.sequence_input_b,embed_w_r_b,embed_h_r_b],name='concat_b')
        concat_a = concatenate([self.sequence_input_a,embed_w_r_a,embed_h_r_a], name='concat_a')
        concat_x = concatenate([embed_w_x,embed_h_x], name='concat_x')


        btn_b = BatchNormalization()(concat_b)
        btn_a = BatchNormalization()(concat_a)

        gru_b = GRU(self.params['gru_b'], input_shape=(self.seq,self.features), return_sequences=True, name='gru_before1')(btn_b)
        gru_a = GRU(self.params['gru_a'], input_shape=(self.seq,self.features),go_backwards=True, return_sequences=True, name='gru_after1')(btn_a)

        gru_b = GRU(self.params['gru_b'], input_shape=(self.seq,self.features), return_sequences=True, name='gru_before2')(gru_b)
        gru_a = GRU(self.params['gru_a'], input_shape=(self.seq,self.features), go_backwards=True,return_sequences=True, name='gru_after2')(gru_a)

        att_b = Attention(128, name='Attention_b')([gru_b, self.x_to_be_predicted])
        att_a = Attention(128, name='Attention_a')([gru_a, self.x_to_be_predicted])

        drop_b = Dropout(self.params['drp_b'])(att_b)
        drop_a = Dropout(self.params['drp_a'])(att_a)

        fl1 = drop_b
        fl2 = drop_a

        fl_x = Flatten()(concat_x)

        concat = concatenate([self.sequence_input_stat_a,fl1,self.sequence_input_stat_b,fl2,self.x_to_be_predicted,fl_x])

        btn = BatchNormalization()(concat)

        fc1 = Dense(self.params['dense1'], activation='relu')(btn)

        drop = Dropout(0.15)(fc1)

        fc2 = Dense(self.params['dense2'], activation='relu')(drop)

        full = Dense(self.params['classes'], activation='softmax')(fc2)
        
        model = Model([self.sequence_input_b,self.inputs_w_b,self.inputs_h_b,
                       self.sequence_input_a,self.inputs_w_a,self.inputs_h_a,
                       self.x_to_be_predicted,self.inputs_w_x,self.inputs_h_x,
                       self.sequence_input_stat_b,
                       self.sequence_input_stat_a], full)
        
        adam = optimizers.Adam(lr=self.params['lr'])
        
        model.compile(loss=Losses().focal_loss(),
              optimizer='adam',
              metrics=['acc'])
        
        model.summary()
        
        return model

    def fit_model(self, model, get_batches_train, get_batches_val):
        
        checkpoint = ModelCheckpoint(self.params['model_name'], 
                                 monitor='val_acc',
                                 verbose=1, 
                                 save_best_only=True,
                                 mode='max')
        
        callbacks_list = [checkpoint]
        
        result = model.fit(x=get_batches_train[0],
                           y=get_batches_train[1], 
                           batch_size=self.params['batch'], 
                           epochs=self.params['epochs'],
                           verbose=1, 
                           validation_data=get_batches_val,
                           callbacks=callbacks_list)
            
        return result, model
    
    def predict(self, model, data):
        return model.predict(data, verbose=1)
    
    def load_model(self, model, model_name):
        model.load_weights(model_name)
        return
    
    def get_embeddings(self, layer_name, model_name, model,sample):
        
        model.load_weights(model_name)
        
                
        embedding_model = Model([self.sequence_input_b,self.inputs_w_b,self.inputs_h_b,
                       self.sequence_input_a,self.inputs_w_a,self.inputs_h_a,
                       self.x_to_be_predicted,self.inputs_w_x,self.inputs_h_x,
                       self.sequence_input_stat_b,
                       self.sequence_input_stat_a],
                        model.get_layer(layer_name).output)
        
        return embedding_model.predict(x=sample, verbose=1)
        
    

    def inverse_transform(self, data):
        classes = list()
        for item in tqdm(data):
            label_idx = np.argmax(item, axis=0).tolist()
            classes.append(label_idx)
        return classes
    
    def plot_confusion_matrix(self, y_test, y_predicted):
        matrix = confusion_matrix(y_test, y_predicted)
        classes = np.unique(y_predicted)
        def plot(cm, classes,
                  normalize=False,
                  title='Confusion matrix',
                  cmap=plt.cm.Blues):
            """
            This function prints and plots the confusion matrix.
            Normalization can be applied by setting `normalize=True`.
            """
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                print("Normalized confusion matrix")
            else:
                print('Confusion matrix, without normalization')

            print(cm)

            plt.imshow(cm, interpolation='nearest', cmap=cmap)
            plt.title(title)
            plt.colorbar()
            tick_marks = np.arange(len(classes))
            plt.xticks(tick_marks, classes, rotation=45)
            plt.yticks(tick_marks, classes)

            fmt = '.2f' if normalize else 'd'
            thresh = cm.max() / 2.
            for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
                plt.text(j, i, format(cm[i, j], fmt),
                         horizontalalignment="center",
                         color="white" if cm[i, j] > thresh else "black")

            plt.ylabel('True label')
            plt.xlabel('Predicted label')
            plt.tight_layout()
        return plot(matrix, classes)
    
    def print_scores(self, y_test, predicted_list):
        from sklearn_crfsuite import scorers
        from sklearn_crfsuite import metrics
        from sklearn.metrics import make_scorer

        print(metrics.flat_classification_report(
            y_test.astype(str), np.array(predicted_list).astype(str), labels=['0','1','2','3'], digits=3
        ))



