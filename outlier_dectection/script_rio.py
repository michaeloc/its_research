import numpy as np
from sklearn.model_selection import train_test_split
import h5py
import rio

# --------Recife---------------------------------------------------
# data_path = './data_recife_uber_pac_lat_lng_timestamp_idtraj_idpoint_idnoise.hdf5'
# params = {'conv1':128,
#           'conv2':64,
#           'conv3':64,
#           'conv4':128,
#           'conv5':128,
#           'conv6':128,
#           'd1':500,
#           'd2':500,
#           'lr':0.01,
#           'op':'adam',
#           'batch':128,
#           'classes':82,
#           'epochs':100,
#           'model_name':'rio_model_clean_data_recife.hdf5'
# }
# ----------end Recife----------------------------------------------
#--------Dublin-----------------------------------------------------
data_path = './data_dublin_uber_pac_lat_lng_timestamp_idtraj_idpoint_idnoise.hdf5'
params = {'conv1':32,
          'conv2':128,
          'conv3':64,
          'conv4':128,
          'conv5':64,
          'conv6':256,
          'd1':1000,
          'd2':500,
          'lr':0.1,
          'op':'adam',
          'batch':128,
          'classes':66,
          'epochs':100,
          'model_name':'rio_model_clean_data_dublin.hdf5'
}


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

    x_train = X_train[:,:,[50,51,52]]
    x_val = X_val[:,:,[50,51,52]]
    x_test = X_test[:,:,[50,51,52]]
    
    return x_train,y_train,x_val,y_val,x_test,y_test

X_train, Y_train,X_val,Y_val, X_test, Y_test = data()

inputs = {'x_train':X_train,
          'y_train':Y_train,
          'x_val':X_val,
          'y_val':Y_val,
          'x_test':X_test,
          'y_test':Y_test
         }

c_rio = rio.Rio(params)
model_rio = c_rio.build_model()
print(model_rio.summary())
hist, model = c_rio.fit_model(model_rio, inputs)
c_rio.print_scores(model, inputs)

