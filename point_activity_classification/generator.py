import numpy as np
import tensorflow as tf
from tqdm import tqdm

class Generator():   
    
    def get_batch(self, x, y):

        list_x_b, list_x_a = list(), list()
        list_x_target, list_stat_b, list_stat_a = list(), list(), list()

        x1 = x
        y1 = tf.one_hot(y,4)
        y1 = tf.reshape(y1,(len(y),-1)).numpy()

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
        print(a_x_b.shape)
        
        return [a_x_b[:,:,[0,1,2,3,4,5,6,9]],a_x_b[:,:,7].astype(int),a_x_b[:,:,8].astype(int),
                a_x_a[:,:,[0,1,2,3,4,5,6,9]],a_x_a[:,:,7].astype(int),a_x_a[:,:,8].astype(int),
                x_t[:,[0,1,2,3,4,5,6,10]], x_t[:,8], x_t[:,9],
                stat_b,
                stat_a],y1,x_t[:,:]
# [0,1,2,3,4,5,6,9]
# x_t no final para pegar o id de cada ponto e construir a matriz
    