#%%
import os
import sys


import math
import time
import argparse
import collections
import logging
import pathlib
import re
import string

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import tensorflow as tf
import h5py
from tqdm import tqdm
from h3 import h3
import copy
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve,auc, precision_score, recall_score

from transformer_model import Transformer, build_model, train_model_builded, evaluate_model, save_model, load_model

#%%
def define_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default="train",
                        help='choose a mode: train or eval')
    parser.add_argument('--source', type=str, default="recife",
                        help='soure of database')
                        
    parser.add_argument('--data_filename', type=str, default='dublin_data_from_pipeline/processed_dublin_train.csv',#'porto_icde/processed_dublin_test_and_anom_r32.csv',
                        help='data file')
    parser.add_argument('--sink', type=str, default='transformer_evaluation/')
    parser.add_argument('--map_size', type=tuple, default=15000,
                        help='number of different point (map)')
    parser.add_argument('--d_model', type=int, default=256,
                        help='size of input embedding')
    parser.add_argument('--num_layers', type=int, default=4,
                        help='size of transform layer')
    parser.add_argument('--num_heads', type=int, default=8,
                        help='size of head attention')
    parser.add_argument('--model_dir', type=str, default="./checkpoints/tranformer_new_version_dublin",#"./checkpoints/tranformer_new_version_recife",
                        help='model dir')
    parser.add_argument('--num_epochs', type=int, default=30,
                        help='number of epochs')
    # parser.add_argument('--learning_rate', type=float, default=0.001,
    #                     help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='minibatch size')
    parser.add_argument('--optimizer', type=str, default="adam",
                        help='optimizer')
    args = parser.parse_args()
    return args
#%%
# class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
#   def __init__(self, d_model, warmup_steps=4000):
#     super(CustomSchedule, self).__init__()

#     self.d_model = d_model
#     self.d_model = tf.cast(self.d_model, tf.float32)

#     self.warmup_steps = warmup_steps

#   def __call__(self, step):
#     arg1 = tf.math.rsqrt(step)
#     arg2 = step * (self.warmup_steps ** -1.5)

#     return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# %%
# def loss_function(real, pred):
#   mask = tf.math.logical_not(tf.math.equal(real, 0))
#   loss_ = loss_object(real, pred)

#   mask = tf.cast(mask, dtype=loss_.dtype)
#   loss_ *= mask

#   return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# %%
def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)

# %%

# def create_padding_mask(seq):
#   seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

#   return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)

# def create_look_ahead_mask(size):
#   mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
#   return mask  # (seq_len, seq_len)

# def create_masks(inp, tar):
#   # Encoder padding mask
#   enc_padding_mask = create_padding_mask(inp)

#   # Used in the 2nd attention block in the decoder.
#   # This padding mask is used to mask the encoder outputs.
#   dec_padding_mask = create_padding_mask(inp)

#   # Used in the 1st attention block in the decoder.
#   # It is used to pad and mask future tokens in the input received by
#   # the decoder.
#   look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
#   dec_target_padding_mask = create_padding_mask(tar)
#   combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

#   return enc_padding_mask, combined_mask, dec_padding_mask

# %%

# train_step_signature = [
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     tf.TensorSpec(shape=(None, None), dtype=tf.int64),
#     ]

# @tf.function(input_signature=train_step_signature)
# def train_step(inp, tar_val,tar_final):
#   tar_inp = tar_val
#   tar_real = tar_final

#   enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

#   with tf.GradientTape() as tape:
#     predictions,_ = transformer(inp, tar_inp,
#                                  True,
#                                  enc_padding_mask,
#                                  combined_mask,
#                                  dec_padding_mask)
    
#     # kl_loss = -0.5 * tf.reduce_mean(s - tf.square(m) - tf.exp(s) + 1)
#     loss = loss_function(tar_real, predictions)
#     # loss+=kl_loss

#   gradients = tape.gradient(loss, transformer.trainable_variables)
#   optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
   
#   train_loss(loss)
#   train_accuracy(accuracy_function(tar_real, predictions))

#%%
# def train_model(input_e_train, input_d_train, target_train):
#     print('Começando o treino :)',flush=True)
#     BUFFER_SIZE = len(input_e_train)
#     BATCH_SIZE = 64
#     steps_per_epoch = len(input_e_train)//BATCH_SIZE
#     for epoch in range(EPOCHS):
#         start = time.time()

#         train_loss.reset_states()
#         train_accuracy.reset_states()
#         e = get_batches(input_e_train,64)
#         d = get_batches(input_d_train,64)
#         t = get_batches(target_train,64)

#         for i in range(steps_per_epoch):
#             inp = next(e)
#             tar_val = next(d)
#             tar = next(t)
#             train_step(inp, tar_val, tar)

#             if i % 50 == 0:
#                 print(f'Epoch {epoch + 1} Batch {i} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

#         if (epoch + 1) % 5 == 0:
#             ckpt_save_path = ckpt_manager.save()
#             print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}',flush=True)

#         print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}',flush=True)

#         print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n',flush=True)
#%%
def evaluate(sentence, max_length=40):
  
  encoder_input = sentence
  
  start, end = 0, 0
  output = tf.convert_to_tensor([start], dtype=tf.int64)
  output = tf.expand_dims(output, 0)
  predictions_scores = list()
  for i in range(max_length):
    enc_padding_mask, combined_mask, dec_padding_mask = create_masks(
        encoder_input, output)

    # predictions.shape == (batch_size, seq_len, vocab_size)
    predictions, attention_weights  = transformer(encoder_input,
                                                 output,
                                                 False,
                                                 enc_padding_mask,
                                                 combined_mask,
                                                 dec_padding_mask)

    # select the last word from the seq_len dimension
    predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

    predicted_id = tf.argmax(predictions, axis=-1)
    # predictions_scores.append(predictions.numpy().reshape(-1)[predicted_id[0][0]])
    predictions_scores.append(predictions)
    
    # print(entropy(predictions.numpy().reshape(-1),base=15000))
    # concatentate the predicted_id to the output which is given to the decoder
    # as its input.
    output = tf.concat([output, predicted_id], axis=-1)

    # return the result if the predicted_id is equal to the end token
    if predicted_id == end:
      break

  # output.shape (1, tokens)
  return output, attention_weights, predictions_scores
#%%
def get_batches(data,batch_size):
    for window in range(0,data.shape[0],batch_size):
        yield data[window:window+batch_size]

#%%
def load_evaluate_data(path):
    trajectories = list()
    for i, eachlines in enumerate(open(path, 'r').readlines()):
        trajectories.append(eval(eachlines))
    trajectories = tf.keras.preprocessing.sequence.pad_sequences(trajectories, maxlen=100, padding='post')  # Colocando o maxlen para 100 recife, 208 pra dublin
    return trajectories
# %%
def get_accuracy_infer(input_test, model):
  accuracies = list()
  outputs = list()
  size = input_test.shape[1]
  atte = None
  preds = None
  for i in tqdm(range(len(input_test))):
    out, atte, preds = model(input_test[i:i+1])
    accuracies.append(accuracy_function(input_test[i:i+1],np.array(preds).reshape((1,size,-1))))
    outputs.append(out)
    
  return accuracies, atte, preds, outputs

def get_metrics_to_auc(scores):
    df_rank = pd.DataFrame()
    df_rank['predicted']=[x.numpy() for x in scores[0]]
    half = len(df_rank)//2
    df_rank['y_true']=[1]*half+[0]*half
    # df_rank = df_rank.sort_values(by='predicted')
    prec, rec, thresh = precision_recall_curve(df_rank.y_true, df_rank.predicted)
    return prec, rec, thresh

def get_region(scores, trajectories):
    pred_region_real = list()
    for i, item in enumerate(scores[3]):
        pred_real = scores[3][i].numpy().reshape(-1)[1:]
        real = trajectories[i]

        
        compare_real = pred_real == real

        pred_region_real.append(np.argwhere(compare_real == False).reshape(-1).tolist())
    
    return pred_region_real

def get_region_metrics(scores, trajectories):
    region_acc = 0
    recall = 0
    precision = 0
    region_true = list()
    region_pred = list()
    half = len(scores[3])//2  # the first half of data doesn't have anomaly
    total = len(scores[3])
    pred_region_real = list()
    pred_region_anom = list()
    real_idx = list()
    anom_idx = list()
    for i, item in enumerate(scores[3]):
        if i < half:
            pred_anom = scores[3][i+half].numpy().reshape(-1)[1:]
            pred_real = scores[3][i].numpy().reshape(-1)[1:]

            anom = trajectories[i+half]
            real = trajectories[i]

            ground_thruth_real = real == real
            ground_thruth_anom = real == anom

            compare_real = pred_real == real
            compare_anom = pred_anom == anom  # antes era real == pred_anom

            pred_region_real.append(np.argwhere(compare_real == False).reshape(-1).tolist())
            pred_region_anom.append(np.argwhere(compare_anom == False).reshape(-1).tolist())

            real_idx.append(np.argwhere(ground_thruth_real == False).reshape(-1).tolist())
            anom_idx.append(np.argwhere(ground_thruth_anom == False).reshape(-1).tolist())
            
            recall += recall_score(ground_thruth_real, compare_real)
            recall += recall_score(ground_thruth_anom, compare_anom)

            precision += precision_score(ground_thruth_real, compare_real)
            precision += precision_score(ground_thruth_anom, compare_anom)
    
    return [precision/total, recall/total, pred_region_real + pred_region_anom, real_idx + anom_idx]

def evaluate_routes(file_path, route, model)-> list:
    current_data = load_evaluate_data(file_path)
    scores = get_accuracy_infer(current_data, model)
    prec, rec, thresh = get_metrics_to_auc(scores)
    print(f'AUC:{auc(rec,prec)}\n')
    num = 2 * rec * prec
    den = rec + prec
    f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den!=0))
    max_f1 = np.max(f1_scores)
    max_f1_thresh = thresh[np.argmax(f1_scores)]
    print(f'max_f1:{max_f1}\t tresh:{thresh}\t max_f1_thresh:{max_f1_thresh}\n')
    average, recall, precision = get_region_metrics(scores, current_data)
    print(f'recall area:{recall}\tprecision area:{precision}')
    return [route, auc(rec,prec), thresh, max_f1, max_f1_thresh, average, recall, precision]

def get_model_output(file_path, route, model)->list:
    current_data = load_evaluate_data(file_path)
    scores = get_accuracy_infer(current_data, model)
    return scores, current_data

def inverse_transform(s, data_anom):
    data_anom = (data_anom[:50,:,[2,3]]*(np.sqrt(s.var_[:2])))+s.mean_[:2]
    return data_anom

def create_data_frame(traj_lat_lng, scores, tokens,routes):
    results = {'id':[],
                    'predicted':[],
                    'input':[],
                    'real_traj':[]
        }
        
    for j in range(100):               
        results['id'] += [j]
        results['real_traj'] += traj_lat_lng[j:j+1,:,:].tolist() 
        results['predicted'] += scores[3][j].numpy()[:,1:].tolist()
        results['input'] += tokens[j].reshape((1,-1)).tolist() 
    
    df = pd.DataFrame(results)
    df['routes'] = [routes]*100
    return df
#%%
if __name__ == '__main__':
    args = define_args()
            
    # learning_rate = CustomSchedule(args.d_model)

    # optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
    #                                  epsilon=1e-9)
    # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    #     from_logits=True, reduction='none')
    
    # train_loss = tf.keras.metrics.Mean(name='train_loss')
    # train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    # dff = 2048
    # dropout_rate = 0.1

    # transformer = Transformer(
    #     num_layers=args.num_layers,
    #     d_model=args.d_model,
    #     num_heads=args.num_heads,
    #     dff=dff,
    #     input_vocab_size=args.map_size,#len(lang_tokenizer.word_index)+1,
    #     target_vocab_size=args.map_size,#len(lang_tokenizer.word_index)+1,
    #     pe_input=10000,
    #     pe_target=10000,
    #     rate=dropout_rate)
    
    # checkpoint_path = args.model_dir

    # ckpt = tf.train.Checkpoint(transformer=transformer,
    #                        optimizer=optimizer)

    # ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # # if a checkpoint exists, restore the latest checkpoint.
    # if ckpt_manager.latest_checkpoint:
    #     ckpt.restore(ckpt_manager.latest_checkpoint)
    #     print('Latest checkpoint restored!!')    

    # EPOCHS = args.num_epochs

    if args.mode == 'train':
        # Mudei aqui para carregar os dados que agora estão tudo com np.array e não csv
        # trajectories = load_evaluate_data(args.data_filename)
        # input_e_train = np.array(trajectories)
        input_e_train = np.load(args.data_filename, allow_pickle=True)[:,:,5].reshape((-1,100)) #208 to dublin, 100 to recife
        zeros = np.zeros((input_e_train.shape[0],1))
        input_d_train = np.concatenate([zeros,input_e_train],axis=1)
        target_train = np.concatenate([input_e_train, zeros],axis=1)

        print(f'{input_e_train[0]}',flush=True)
        print(f'{input_d_train[0]}',flush=True)
        print(f'{target_train[0]}', flush=True)

        transformer = build_model(args.num_epochs,args.num_layers, 
                                args.d_model, args.num_heads, 2048, 
                                0.1, args.model_dir)
        train_model_builded(input_e_train,input_d_train,target_train)
        # train_model(input_e_train,input_d_train,target_train)
        # run_training(args.num_epochs,args.num_layers, 
        #             args.d_model, args.num_heads, 2048, 
        #             0.1, args.model_dir,
        #             input_e_train,input_d_train,target_train)
        print('trainamento')    
    elif args.mode == 'eval':
        evaluate_data = load_evaluate_data(args.data_filename)
        scores = get_accuracy_infer(evaluate_data)
        prec, rec, thresh = get_metrics_to_auc(scores)
        plt.plot(prec,rec)
        print(f'AUC:{auc(rec,prec)}\n')
        num = 2 * rec * prec
        den = rec + prec
        f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresh[np.argmax(f1_scores)]
        print(f'max_f1:{max_f1}\t tresh:{thresh}\t max_f1_thresh:{max_f1_thresh}\n')
        get_region_metrics(scores, evaluate_data)
    elif args.mode == 'eval_loop':
        list_results = list()
        
        transformer = build_model(args.num_epochs,args.num_layers, 
                                args.d_model, args.num_heads, 2048, 
                                0.1, args.model_dir)
        
        model = evaluate_model(transformer, 208)  #sentences size (170 recife and 208 dublin)
        
        for routes in tqdm(os.listdir(args.data_filename)):
            file_name = args.data_filename+routes+f'/processed_{args.source}_test_and_anom_evaluation.csv'
            list_results.append(evaluate_routes(file_name,routes,model))
        
        df_results = pd.DataFrame(list_results, columns=['Rota','Auc','Thresh','Max F1','F1 Thresh','Intersection','Recall','Precision'])
        
        if not os.path.exists(args.sink):
            os.makedirs(args.sink)

        df_results.to_csv(args.sink+f'transformer_results_{args.source}.csv')
    
    elif args.mode == 'app':
        
        scaler_filename = '../dublin_scaler.save' if args.source == 'dublin' else '../recife_scaler.save'
        s = joblib.load(scaler_filename)
        df = pd.DataFrame()
        
        transformer = build_model(args.num_epochs,args.num_layers, 
                                args.d_model, args.num_heads, 2048, 
                                0.1, args.model_dir)
        
        model = evaluate_model(transformer, 100)  #sentences size (100 recife and 208 dublin)
        list_results = None
        list_traj = list()
        columns=['Rota','Scores','AUC','Thresh','Max F1','Score_Prec','Score_Rec','Intersec_Prec', 'Intersec_Rec','Intersec_Pred_Idx','Intersec_True_Idx']
        df = pd.DataFrame(columns=columns)
        for i, routes in enumerate(tqdm(os.listdir(args.data_filename))):
            file_name = args.data_filename+routes+f'/test_anom_1_03.csv'
            scores, tokens = get_model_output(file_name,routes,model)
            # list_traj.append(np.load(args.data_filename+routes+'/test_anom_lat_lng_1_01.npy').tolist())            
            prec, rec, thresh = get_metrics_to_auc(scores)
            print(f'AUC:{auc(rec,prec)}\n', flush=True)
            num = 2 * rec * prec
            den = rec + prec
            f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den!=0))
            max_f1 = np.max(f1_scores)
            max_f1_thresh = thresh[np.argmax(f1_scores)]
            results = get_region_metrics(scores,tokens)
            list_results = [routes, [x.numpy().tolist() for x in scores[0]], auc(rec,prec),thresh, max_f1, prec.tolist(), rec.tolist()]+results
            df1 = pd.DataFrame([list_results], columns=columns)
            df = pd.concat([df,df1])
            df.to_csv('transformer_evaluation/dataframe_app_recife_1_03.csv')
            # break        
        # if not os.path.exists(args.sink):
        #     os.makedirs(args.sink)            
            
        # df1.to_csv('transformer_evaluation/dataframe_app_dublin.csv')
    
    elif args.mode == 'case_study':
        print('Rodando Caso de estudo BABY')
        scaler_filename = '../dublin_scaler.save' if args.source == 'dublin' else '../recife_scaler.save'
        s = joblib.load(scaler_filename)
        df = pd.DataFrame()
        
        transformer = build_model(args.num_epochs,args.num_layers, 
                                args.d_model, args.num_heads, 2048, 
                                0.1, args.model_dir)
        
        model = evaluate_model(transformer, 208)  #sentences size (100 recife and 208 dublin)
        list_results = None
        list_traj = list()
        columns=['Rota','Scores','Predicted_Tokens','Real_Tokens']
        df = pd.DataFrame(columns=columns)
        
        x = np.load(args.data_filename+f'/X_test_rearranged.npy', allow_pickle=True)
        y = np.load(args.data_filename+f'/y_test_rearranged.npy', allow_pickle=True)
        unique_y = np.unique(np.argmax(y,axis=1))
        for item in unique_y:
            idx = np.argwhere(np.argmax(y,axis=1) == item)
            trajectories = x[idx,:,5].reshape((-1,208))
            print(f'Shape of traj:{trajectories.shape}')
            scores = get_accuracy_infer(trajectories, model)                        
            # prec, rec, thresh = get_metrics_to_auc(scores)
            # print(f'AUC:{auc(rec,prec)}\n', flush=True)
            
            # num = 2 * rec * prec
            # den = rec + prec
            # f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den!=0))
            # max_f1 = np.max(f1_scores)
            # max_f1_thresh = thresh[np.argmax(f1_scores)]

            routes = item            
            
            data_dict = {
                'Rota': routes,
                'Scores':[x.numpy().tolist() for x in scores[0]],
                'Predicted_Tokens':[x.numpy().tolist() for x in scores[3]],
                'Real_Tokens':[x.tolist() for x in trajectories]
            }
            df1 = pd.DataFrame(data_dict)
            df = pd.concat([df,df1])
            df.to_csv('transformer_evaluation/case_study_dub.csv',index=False)
            
    else:
        raise ValueError("Mode not in ['pretrain', 'train', 'eval'].")