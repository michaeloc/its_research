import tensorflow as tf
from keras import backend as K
from keras import regularizers, constraints, initializers, activations
from keras.layers import Dense
from keras.engine import InputSpec
from keras.layers import Layer
class Attention(Layer):

    def __init__(self, units,
                 activation='tanh',
                 name='Attention',
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):

        self.units = units
        self.activation = activations.get(activation)
        self.kernel_initializer = initializers.get(kernel_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        super(Attention, self).__init__(**kwargs)
        self.name = name

    def build(self, input_shape):

        self.batch_size, self.timesteps, self.input_dim = input_shape[0]
        self.batch_size1, self.input_dim1 = input_shape[1]


        self.states = None

        """
            Matrices for creating the context vector
        """

        self.V_a = self.add_weight(shape=(self.units,),
                                   name='V_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.W_u = self.add_weight(shape=(self.input_dim1, self.units),
                                   name='W_u',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.U_a = self.add_weight(shape=(self.input_dim, self.units),
                                   name='U_a',
                                   initializer=self.kernel_initializer,
                                   regularizer=self.kernel_regularizer,
                                   constraint=self.kernel_constraint,
                                   trainable=True)
        self.b_a = self.add_weight(shape=(self.units,),
                                   name='b_a',
                                   initializer=self.bias_initializer,
                                   regularizer=self.bias_regularizer,
                                   constraint=self.bias_constraint,
                                   trainable=True)
        

        self.input_spec = [
            InputSpec(shape=(self.batch_size, self.timesteps, self.input_dim)),
            InputSpec(shape=(self.batch_size1, self.input_dim1))]
        

        super(Attention, self).build(input_shape)


    def call(self, x):
        h_seq = x[0]
        #W*h + b
        h = K.reshape(h_seq, (-1, self.input_dim))
        h = K.dot(h, self.U_a)
        h = K.bias_add(h, self.b_a)
        h = K.reshape(h, K.stack([-1, self.timesteps, self.units]))
        h.set_shape([None, None, self.units])
        
        #u_w - context vector
        stm = x[1]
        _stm = K.repeat(stm, self.timesteps)

        # now multiplty the weight matrix with the repeated hidden state
        _Wxstm = K.dot(_stm, self.W_u)

        # tan        
        et = K.dot(activations.tanh(_Wxstm + h),
                   K.expand_dims(self.V_a))
        #softmax
        at = K.exp(et)
        at_sum = K.sum(at, axis=1)
        at_sum_repeated = K.repeat(at_sum, self.timesteps)
        at /= at_sum_repeated  # vector of size (batchsize, timesteps, 1)

        # calculate the context vector
        context = K.squeeze(K.batch_dot(at, h_seq, axes=1), axis=1)
        
        return context

    def compute_output_shape(self, input_shape):
        """
            For Keras internal compatability checking
        """

        return (None, self.units)
    
    def get_config(self):
        """
            For rebuilding models on load time.
        """
        config = {
            'output_dim': self.units,
            'units': self.units
        }
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))