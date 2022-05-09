
import collections
import logging
import os
import pathlib
import re
import string
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf
import h5py
from tqdm import tqdm
from h3 import h3
import copy
from scipy.stats import entropy
from sklearn.metrics import precision_recall_curve,auc

EPOCHS = 50
#%%
# testing the reparametrization for future implementation of variational autoencoder
class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim1 = tf.shape(z_mean)[1]
        dim2 = tf.shape(z_mean)[2]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim1, dim2))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# %%
def get_angles(pos, i, d_model):
  angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
  return pos * angle_rates
# %%
def positional_encoding(position, d_model):
  angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                          np.arange(d_model)[np.newaxis, :],
                          d_model)

  # apply sin to even indices in the array; 2i
  angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

  # apply cos to odd indices in the array; 2i+1
  angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

  pos_encoding = angle_rads[np.newaxis, ...]

  return tf.cast(pos_encoding, dtype=tf.float32)
# %%
def create_padding_mask(seq):
  seq = tf.cast(tf.math.equal(seq, 0), tf.float32)

  return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)
# %%
def create_look_ahead_mask(size):
  mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
  return mask  # (seq_len, seq_len)
# %%
def scaled_dot_product_attention(q, k, v, mask):
  """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

  matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

  # scale matmul_qk
  dk = tf.cast(tf.shape(k)[-1], tf.float32)
  scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

  # add the mask to the scaled tensor.
  if mask is not None:
    scaled_attention_logits += (mask * -1e9)

  # softmax is normalized on the last axis (seq_len_k) so that the scores
  # add up to 1.
  attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)

  output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

  return output, attention_weights
# %%
class MultiHeadAttention(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads):
    super(MultiHeadAttention, self).__init__()
    self.num_heads = num_heads
    self.d_model = d_model

    assert d_model % self.num_heads == 0

    self.depth = d_model // self.num_heads

    self.wq = tf.keras.layers.Dense(d_model)
    self.wk = tf.keras.layers.Dense(d_model)
    self.wv = tf.keras.layers.Dense(d_model)

    self.dense = tf.keras.layers.Dense(d_model)

  def split_heads(self, x, batch_size):
    """Split the last dimension into (num_heads, depth).
    Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
    """
    x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
    return tf.transpose(x, perm=[0, 2, 1, 3])

  def call(self, v, k, q, mask):
    batch_size = tf.shape(q)[0]

    q = self.wq(q)  # (batch_size, seq_len, d_model)
    k = self.wk(k)  # (batch_size, seq_len, d_model)
    v = self.wv(v)  # (batch_size, seq_len, d_model)

    q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
    k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
    v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

    # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
    # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
    scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)

    scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

    concat_attention = tf.reshape(scaled_attention,
                                  (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

    output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

    return output, attention_weights
# %%
def point_wise_feed_forward_network(d_model, dff):
  return tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
      tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
  ])
# %%
class EncoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(EncoderLayer, self).__init__()

    self.mha = MultiHeadAttention(d_model, num_heads)
    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
    attn_output = self.dropout1(attn_output, training=training)
    out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

    ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
    ffn_output = self.dropout2(ffn_output, training=training)
    out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

    return out2
# %%
class DecoderLayer(tf.keras.layers.Layer):
  def __init__(self, d_model, num_heads, dff, rate=0.1):
    super(DecoderLayer, self).__init__()

    self.mha1 = MultiHeadAttention(d_model, num_heads)
    self.mha2 = MultiHeadAttention(d_model, num_heads)

    self.ffn = point_wise_feed_forward_network(d_model, dff)

    self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
    self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    self.dropout1 = tf.keras.layers.Dropout(rate)
    self.dropout2 = tf.keras.layers.Dropout(rate)
    self.dropout3 = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):
    # enc_output.shape == (batch_size, input_seq_len, d_model)

    attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
    attn1 = self.dropout1(attn1, training=training)
    out1 = self.layernorm1(attn1 + x)

    attn2, attn_weights_block2 = self.mha2(
        enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
    attn2 = self.dropout2(attn2, training=training)
    out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

    ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
    ffn_output = self.dropout3(ffn_output, training=training)
    out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

    return out3, attn_weights_block1, attn_weights_block2
# %%
class Encoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Encoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(input_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding,
                                            self.d_model)

    self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]

    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, training, mask):

    seq_len = tf.shape(x)[1]

    # adding embedding and position encoding.
    x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x = self.enc_layers[i](x, training, mask)

    return x  # (batch_size, input_seq_len, d_model)
# %%
class Decoder(tf.keras.layers.Layer):
  def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
               maximum_position_encoding, rate=0.1):
    super(Decoder, self).__init__()

    self.d_model = d_model
    self.num_layers = num_layers

    self.embedding = tf.keras.layers.Embedding(target_vocab_size, d_model)
    self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

    self.dec_layers = [DecoderLayer(d_model, num_heads, dff, rate)
                       for _ in range(num_layers)]
    self.dropout = tf.keras.layers.Dropout(rate)

  def call(self, x, enc_output, training,
           look_ahead_mask, padding_mask):

    seq_len = tf.shape(x)[1]
    attention_weights = {}

    x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x += self.pos_encoding[:, :seq_len, :]

    x = self.dropout(x, training=training)

    for i in range(self.num_layers):
      x, block1, block2 = self.dec_layers[i](x, enc_output, training,
                                             look_ahead_mask, padding_mask)

      attention_weights[f'decoder_layer{i+1}_block1'] = block1
      attention_weights[f'decoder_layer{i+1}_block2'] = block2

    # x.shape == (batch_size, target_seq_len, d_model)
    return x, attention_weights

class Transformer(tf.keras.Model):  ## New class updated in order to save the model
  def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
               target_vocab_size, pe_input, pe_target, rate=0.1):
    super(Transformer, self).__init__()

    self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
                             input_vocab_size, pe_input, rate)

    self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                           target_vocab_size, pe_target, rate)
    
    self.final_layer = tf.keras.layers.Dense(target_vocab_size)

  def call(self, inputs, training):
    inp, tar = inputs
    enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)
    enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    dec_output, attention_weights = self.decoder(
        tar, enc_output, training, look_ahead_mask, dec_padding_mask)
    
    final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)
    return final_output, attention_weights
  
  def create_masks(self, inp, tar):
    # Encoder padding mask
    enc_padding_mask = create_padding_mask(inp)

    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = create_padding_mask(inp)

    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by
    # the decoder.
    look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = create_padding_mask(tar)
    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask
# %%
# class Transformer(tf.keras.Model):
#   def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
#                target_vocab_size, pe_input, pe_target, rate=0.1):
#     super(Transformer, self).__init__()

#     self.tokenizer = Encoder(num_layers, d_model, num_heads, dff,
#                              input_vocab_size, pe_input, rate)

#     self.decoder = Decoder(num_layers, d_model, num_heads, dff,
#                            target_vocab_size, pe_target, rate)
    
#     self.final_layer = tf.keras.layers.Dense(target_vocab_size)

#     # self.mean = tf.keras.layers.Dense(d_model)
#     # self.deviation = tf.keras.layers.Dense(d_model)
#     # self.sampling = Sampling()
    

#   def call(self, inp, tar, training, enc_padding_mask,
#            look_ahead_mask, dec_padding_mask):

#     enc_output = self.tokenizer(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)
    
#     # m = self.mean(enc_output)
#     # s = self.deviation(enc_output)
#     # z = self.sampling((m,s))
    
#     # dec_output.shape == (batch_size, tar_seq_len, d_model)
#     dec_output, attention_weights = self.decoder(
#         tar, enc_output, training, look_ahead_mask, dec_padding_mask)

#     final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

#     return final_output, attention_weights#, m, s, z
# %%
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
  def __init__(self, d_model, warmup_steps=4000):
    super(CustomSchedule, self).__init__()

    self.d_model = d_model
    self.d_model = tf.cast(self.d_model, tf.float32)

    self.warmup_steps = warmup_steps

  def __call__(self, step):
    arg1 = tf.math.rsqrt(step)
    arg2 = step * (self.warmup_steps ** -1.5)

    return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

# %%
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
# %%
def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

# %%
def accuracy_function(real, pred):
  accuracies = tf.equal(real, tf.argmax(pred, axis=2))

  mask = tf.math.logical_not(tf.math.equal(real, 0))
  accuracies = tf.math.logical_and(mask, accuracies)

  accuracies = tf.cast(accuracies, dtype=tf.float32)
  mask = tf.cast(mask, dtype=tf.float32)
  return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)
# %%
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

#%%
def get_batches(data,batch_size):
    for window in range(0,data.shape[0],batch_size):
        yield data[window:window+batch_size]

# %%
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
train_step_signature = [
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    tf.TensorSpec(shape=(None, None), dtype=tf.int64),
]


@tf.function(input_signature=train_step_signature)
def train_step(inp, tar_val,tar_final):
  tar_inp = tar_val
  tar_real = tar_final

  # enc_padding_mask, combined_mask, dec_padding_mask = create_masks(inp, tar_inp)

  with tf.GradientTape() as tape:
    predictions,_ = transformer([inp, tar_inp],
                                 True)
    
    # kl_loss = -0.5 * tf.reduce_mean(s - tf.square(m) - tf.exp(s) + 1)
    loss = loss_function(tar_real, predictions)
    # loss+=kl_loss

  gradients = tape.gradient(loss, transformer.trainable_variables)
  optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))
   
  train_loss(loss)
  train_accuracy(accuracy_function(tar_real, predictions))

#%%
def train_model(input_e_train,input_d_train,target_train):
    print('Começando o treino :)',flush=True)
    BUFFER_SIZE = len(input_e_train)
    BATCH_SIZE = 64
    steps_per_epoch = len(input_e_train)//BATCH_SIZE
    for epoch in range(EPOCHS):
      start = time.time()

      train_loss.reset_states()
      train_accuracy.reset_states()
      e = get_batches(input_e_train,64)
      d = get_batches(input_d_train,64)
      t = get_batches(target_train,64)

      for i in range(steps_per_epoch):
          inp = next(e)
          tar_val = next(d)
          tar = next(t)
          train_step(inp, tar_val, tar)

          if i % 50 == 0:
            print(f'Epoch {epoch + 1} Batch {i} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}', flush=True)

      if (epoch + 1) % 5 == 0:
          ckpt_save_path = ckpt_manager.save()
          print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}',flush=True)

      print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}',flush=True)

      print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n',flush=True)

#%%
def build_model(epoch, num_layers, d_model, num_heads, dff, dropout_rate, ckpt_path):
  
  global transformer 
  transformer = Transformer(
    num_layers=num_layers,
    d_model=d_model,
    num_heads=num_heads,
    dff=dff,
    input_vocab_size=15000,#len(lang_tokenizer.word_index)+1,
    target_vocab_size=15000,#len(lang_tokenizer.word_index)+1,
    pe_input=10000,
    pe_target=10000,
    rate=dropout_rate)

  learning_rate = CustomSchedule(d_model)

  global optimizer 
  optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)
  checkpoint_path = ckpt_path

  ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)
  global ckpt_manager
  ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

  # if a checkpoint exists, restore the latest checkpoint.
  if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!!') 
  return transformer

def train_model_builded(input_e_train,input_d_train,target_train):
  train_model(input_e_train,input_d_train,target_train)
  
# %%
class Evaluate(tf.Module):
  def __init__(self, transformer):
    self.transformer = transformer

  def __call__(self, sentence, max_length=40):
 
    assert isinstance(sentence, tf.Tensor)

    encoder_input = sentence
    
    start, end = 0, 0
    
    start = tf.convert_to_tensor([start], dtype=tf.int64)
    end = tf.convert_to_tensor([end], dtype=tf.int64)
    
    output_array = tf.TensorArray(dtype=tf.int64, size=0, dynamic_size=True)
    output_array = output_array.write(0, start)
    predictions_scores = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
    for i in tf.range(max_length):
      output = tf.transpose(output_array.stack())
    
      predictions, _  = self.transformer([encoder_input,output],False)

    
      predictions = predictions[:, -1:, :]  # (batch_size, 1, vocab_size)

      predicted_id = tf.argmax(predictions, axis=-1)
    
      predictions_scores = predictions_scores.write(i,tf.reshape(predictions,[15000]))
      
      
      output_array = output_array.write(i+1, predicted_id[0])     

    
      if predicted_id == end:
        break
    output = tf.transpose(output_array.stack())
    _, attention_weights = self.transformer([encoder_input, output[:,:-1]], training=False)
    
    return output, attention_weights, predictions_scores.stack()
    # return output_array, output_array, output_array

#%%
class TransformEvaluator(tf.Module):
  def __init__(self, evaluator, max_length=208):
    self.evaluator = evaluator
    self.max_length = max_length

  @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
  def __call__(self, sentence):
    (output,
      attention,
      predictions_scores
    ) = self.evaluator(sentence, max_length=self.max_length)
    return output, attention, predictions_scores
  
  @tf.function(input_signature=[tf.TensorSpec(shape=(None, None), dtype=tf.int64)])
  def predict(self, sentence):
    (output,
      attention,
      predictions_scores
    ) = self.evaluator(sentence, max_length=self.max_length)
    return output, attention, predictions_scores
#%%
def evaluate_model(transformer, size=208):
  evaluate_m = Evaluate(transformer)
  evaluator = TransformEvaluator(evaluate_m, size)
  # return evaluate_m(sentence,208)
  return evaluator

def save_model(tranformer, max_length, path):
  evaluate_m = Evaluate(transformer)
  evaluator = TransformEvaluator(evaluate_m)
  tf.saved_model.save(evaluator, export_dir = path)

def load_model(path):
  return tf.saved_model.load(path)
# %%
# def load_data():
#     trajectories = list()
#     for i, eachlines in enumerate(open('porto_icde/processed_dublin_test_and_anom_r32.csv', 'r').readlines()):
#         trajectories.append(eval(eachlines))
#     trajectories = tf.keras.preprocessing.sequence.pad_sequences(trajectories, padding='post')
#     return trajectories
# # %%
# def get_accuracy_infer(input_test):
#   accuracies = list()
#   outputs = list()
#   size = input_test.shape[1]
#   atte = None
#   preds = None
#   for i in tqdm(range(len(input_test))):
#     out, atte, preds = evaluate(input_test[i:i+1],size)
#     accuracies.append(accuracy_function(input_test[i:i+1],np.array(preds).reshape((1,size,-1))))
#     outputs.append(out)
#   return accuracies, atte, preds, outputs
# # %%
# scores = get_accuracy_infer(trajectories)
# # %%
# df_rank = pd.DataFrame()
# df_rank['predicted']=[x.numpy() for x in scores[0]]
# df_rank['y_true']=[1]*50+[0]*50
# # df_rank = df_rank.sort_values(by='predicted')
# prec, rec, thresh = precision_recall_curve(df_rank.y_true, df_rank.predicted)
# plt.plot(prec,rec)
# # %%
# auc(rec,prec)
# #%%
# num = 2 * rec * prec
# den = rec + prec
# f1_scores = np.divide(num, den, out=np.zeros_like(den), where=(den!=0))
# max_f1 = np.max(f1_scores)
# max_f1_thresh = thresh[np.argmax(f1_scores)]

# print(f'max_f1:{max_f1}\t tresh:{thresh}\t max_f1_thresh:{max_f1_thresh}')

# #%%
# region_true = list()
# region_pred = list()
# region_acc = 0
# for i, item in enumerate(scores[3]):
#   if i < 50:
#     pred = scores[3][i+50].numpy().reshape(-1)[1:]
#     anom = trajectories[i+50]
#     real = trajectories[i]

#     eq_real_anom = real == anom
#     eq_real_pred = real == pred

#     region_true.append(np.argwhere(eq_real_anom == False).reshape(-1).tolist())
#     region_pred.append(np.argwhere(eq_real_pred == False).reshape(-1).tolist())

#     region_acc += len(set(region_true[i]).intersection(region_pred[i]))/len(region_true[i])
# print(np.mean(region_acc)/50)
  
# # %%
# def plot_attention_head(in_tokens, translated_tokens, attention):
#   # The plot is of the attention when a token was generated.
#   # The model didn't generate `<START>` in the output. Skip it.
#   translated_tokens = translated_tokens.numpy().reshape(-1)[1:31]
#   in_tokens = in_tokens.reshape(-1)[:30]

#   plt.figure(figsize=(20,20))
#   ax = plt.gca()
#   ax.matshow(attention[:30,:30])
#   ax.set_xticks(range(len(in_tokens)))
#   ax.set_yticks(range(len(translated_tokens)))

#   labels = [str(label) for label in in_tokens]
#   ax.set_xticklabels(
#       labels, rotation=90)

#   labels = [str(label) for label in translated_tokens]
#   ax.set_yticklabels(labels)
# # %%
# attention_heads = tf.squeeze(
#   scores[1]['decoder_layer4_block2'], 0)
# attention = attention_heads[0]
# attention.shape
# # %%
# plot_attention_head(trajectories[120:121], scores[3], attention)
# # %%
# %%time
# a,b,c = evaluate(trajectories[0:1],208)
# # %%
# %%time
# accuracy_function(trajectories[0:1],np.array(c).reshape((1,208,-1)))
# # %%
