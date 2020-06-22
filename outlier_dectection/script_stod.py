import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import stod

# --------Recife---------------------------------------------------
# data_path = './data_recife_uber_pac_lat_lng_timestamp_idtraj_idpoint_idnoise.hdf5'
# params = {'gru':128,
#           'd1':1000,
#           'seq':100,
#           'features':50
#           'lr':0.1,
#           'drp':0.25,
#           'op':'rmsprop',
#           'batch':64,
#           'classes':82,
#           'epochs':100,
#           'model_name':'stod_model_clean_data_recife.hdf5'
# }
# ----------end Recife----------------------------------------------
#--------Dublin-----------------------------------------------------
data_path = './data_dublin_uber_pac_lat_lng_timestamp_idtraj_idpoint_idnoise.hdf5'
params = {'gru':64,
          'd1':2000,
          'seq':100,
          'features':50,
          'lr':0.1,
          'drp':0.25,
          'op':'rmsprop',
          'batch':128,
          'classes':66,
          'epochs':100,
          'model_name':'stod_model_clean_data_dublin.hdf5'}
#--------end Dublin-----------------------------------------------------

def data():
    X = None
    y = None
    with h5py.File(data_path, 'r') as hf:
        X = hf["X_embedding"][:]
        y = hf["y_embedding"][:]
        hf.flush()
        hf.close()

    X_train,X_test,y_train,y_test = train_test_split(X,y, test_size=0.2, random_state=1023)
    X_train,X_val,y_train,y_val = train_test_split(X_train,y_train, test_size=0.2, random_state=1023)

    x_train = X_train[:,:,:50]
    x_val = X_val[:,:,:50]
    x_test = X_test[:,:,:50]
    
    return x_train,y_train,x_val,y_val,x_test,y_test

X_train, Y_train,X_val,Y_val, X_test, Y_test = data()

inputs = {'x_train':X_train,
          'y_train':Y_train,
          'x_val':X_val,
          'y_val':Y_val,
          'x_test':X_test,
          'y_test':Y_test
         }

c_stod = stod.Stod(params)
model_stod = c_stod.build_model()
print(model_stod.summary())
hist, model = c_stod.fit_model(model_stod, inputs)
c_stod.print_scores(model, inputs)

